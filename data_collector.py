"""
data_collector.py
=================
Responsible for gathering all raw data used by the pipeline:
  1. Stock price OHLCV data via yfinance
  2. Financial news headlines from Reuters RSS feed
  3. Reddit posts/comments from finance-related subreddits via PRAW
  4. Synthetic tweets (realistic, stock-aware) when Twitter API is unavailable

All public functions return plain Python lists/dicts or pandas DataFrames so
downstream modules stay dependency-free with respect to data-source details.
"""

import datetime
import random
import time
import warnings
import logging

import feedparser       # RSS / Atom feed parsing
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# How far back to pull price history (in calendar days)
PRICE_HISTORY_DAYS = 90

# Reuters RSS URLs to try in order
REUTERS_RSS_URLS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
]

# Reddit subreddits to scrape
REDDIT_SUBREDDITS = ["wallstreetbets", "investing", "stocks"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  STOCK PRICE DATA  (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(
    tickers: list[str] = TICKERS,
    days: int = PRICE_HISTORY_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV (Open, High, Low, Close, Volume) data for each ticker.

    Returns
    -------
    dict mapping ticker -> DataFrame with columns:
        Open, High, Low, Close, Volume, plus computed features:
        MA5 (5-day moving average), MA20, Daily_Return, Volatility
    """
    end   = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    price_data = {}

    for ticker in tickers:
        logger.info(f"Fetching price data for {ticker} ...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                raise ValueError("Empty DataFrame returned")

            # Flatten MultiIndex columns produced by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)

            # ── Technical feature engineering ──────────────────────────────
            df["MA5"]         = df["Close"].rolling(window=5).mean()
            df["MA20"]        = df["Close"].rolling(window=20).mean()
            df["Daily_Return"] = df["Close"].pct_change()          # % change
            df["Volatility"]  = df["Daily_Return"].rolling(5).std()

            # Binary label: 1 = next-day price goes UP, 0 = DOWN / flat
            df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

            # Magnitude of the next-day move (regression target)
            df["Return_Magnitude"] = df["Close"].pct_change().shift(-1)

            df.dropna(inplace=True)
            price_data[ticker] = df
            logger.info(f"  [OK] {ticker}: {len(df)} trading days loaded")

        except Exception as exc:
            logger.warning(f"  [X] {ticker}: yfinance failed ({exc}). Using synthetic prices.")
            price_data[ticker] = _generate_synthetic_prices(ticker, days)

    return price_data


def _generate_synthetic_prices(ticker: str, days: int) -> pd.DataFrame:
    """
    Fallback: create a realistic random-walk OHLCV series when yfinance fails.
    Uses geometric Brownian motion so the series looks like real stock data.
    """
    logger.info(f"    Generating synthetic price data for {ticker} ...")
    np.random.seed(hash(ticker) % 2**31)

    # Starting price - different per ticker so plots look distinct
    start_prices = {"AAPL": 180, "TSLA": 250, "GOOGL": 140, "MSFT": 380, "AMZN": 185}
    S0 = start_prices.get(ticker, 150)

    dates  = pd.bdate_range(end=datetime.date.today(), periods=days)
    mu     = 0.0005          # drift (tiny daily mean return)
    sigma  = 0.018           # daily volatility
    dt     = 1.0

    returns = np.random.normal(mu, sigma, len(dates))
    prices  = S0 * np.exp(np.cumsum(returns))

    # Build OHLCV
    df = pd.DataFrame(index=dates)
    df["Close"]  = prices
    df["Open"]   = df["Close"].shift(1).fillna(prices[0])
    noise        = np.abs(np.random.normal(0, sigma * 0.5, len(dates)))
    df["High"]   = df[["Open", "Close"]].max(axis=1) * (1 + noise)
    df["Low"]    = df[["Open", "Close"]].min(axis=1) * (1 - noise)
    df["Volume"] = np.random.randint(5_000_000, 80_000_000, len(dates))

    df["MA5"]          = df["Close"].rolling(5).mean()
    df["MA20"]         = df["Close"].rolling(20).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(5).std()
    df["Target"]       = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Return_Magnitude"] = df["Close"].pct_change().shift(-1)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NEWS DATA  (Reuters RSS + fallback)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_headlines(max_articles: int = 200) -> list[dict]:
    """
    Try each Reuters / Google News RSS URL in order.
    Returns a list of dicts: {"text": str, "published": datetime, "source": "news"}.
    Falls back to synthetic headlines if all feeds fail.
    """
    articles = []

    for url in REUTERS_RSS_URLS:
        try:
            logger.info(f"Fetching RSS feed: {url}")
            # feedparser honours HTTP redirects and handles SSL gracefully
            feed = feedparser.parse(url)
            if not feed.entries:
                continue

            for entry in feed.entries:
                title   = getattr(entry, "title",   "")
                summary = getattr(entry, "summary", "")
                text    = f"{title}. {summary}".strip()

                # Parse publication time (may be absent in some feeds)
                try:
                    pub = datetime.datetime(*entry.published_parsed[:6])
                except Exception:
                    pub = datetime.datetime.now()

                articles.append({"text": text, "published": pub, "source": "reuters_rss"})

            logger.info(f"  [OK] Fetched {len(articles)} news items from RSS")
            if len(articles) >= max_articles:
                break

        except Exception as exc:
            logger.warning(f"  RSS feed failed ({exc})")

    if not articles:
        logger.warning("All RSS feeds failed - generating synthetic news")
        articles = _generate_synthetic_news(max_articles)

    return articles[:max_articles]


def _generate_synthetic_news(n: int = 200) -> list[dict]:
    """Generate realistic-looking financial news headlines."""
    templates = [
        "{ticker} shares {move} {pct}% after {event}",
        "Analysts {upgrade} {ticker} amid {reason}",
        "{ticker} Q{q} earnings {beat_miss} expectations",
        "Market reacts to {ticker}'s new {product} announcement",
        "Investors {sentiment} about {ticker}'s growth prospects",
        "{ticker} CEO says company is '{outlook}' for the year",
        "Hedge funds {action} positions in {ticker}",
        "Tech stocks {direction} as {macro} pressures mount",
        "{ticker} revenue {up_down} {pct}% year-over-year",
    ]
    moves      = ["surge", "fall", "climb", "dip", "rally", "plummet", "jump", "slide"]
    directions = ["rally", "sell off", "trade mixed", "edge higher", "retreat"]
    events     = ["strong earnings", "product launch", "analyst upgrade", "market optimism",
                  "regulatory approval", "guidance cut", "partnership deal"]
    reasons    = ["strong fundamentals", "AI tailwinds", "cost-cutting measures",
                  "market share gains", "supply chain issues"]
    actions    = ["increase", "decrease", "maintain", "rotate out of", "build up"]
    macros     = ["inflation", "interest rate", "Fed policy", "geopolitical", "recession"]
    sentiments = ["optimistic", "cautious", "bullish", "bearish", "neutral"]
    outlooks   = ["well-positioned", "cautiously optimistic", "on track", "agile"]
    products   = ["AI chip", "cloud service", "EV model", "subscription plan", "streaming feature"]

    articles = []
    now = datetime.datetime.now()
    for i in range(n):
        tmpl   = random.choice(templates)
        ticker = random.choice(TICKERS)
        text   = (
            tmpl
            .replace("{ticker}",    ticker)
            .replace("{move}",      random.choice(moves))
            .replace("{direction}", random.choice(directions))
            .replace("{pct}",       f"{random.uniform(0.5, 8):.1f}")
            .replace("{event}",     random.choice(events))
            .replace("{reason}",    random.choice(reasons))
            .replace("{upgrade}",   random.choice(["upgrade", "downgrade", "maintain rating on"]))
            .replace("{q}",         str(random.randint(1, 4)))
            .replace("{beat_miss}", random.choice(["beat", "missed", "met"]))
            .replace("{product}",   random.choice(products))
            .replace("{sentiment}", random.choice(sentiments))
            .replace("{outlook}",   random.choice(outlooks))
            .replace("{action}",    random.choice(actions))
            .replace("{macro}",     random.choice(macros))
            .replace("{up_down}",   random.choice(["rises", "falls"]))
        )
        pub = now - datetime.timedelta(minutes=random.randint(0, 60 * 24 * 30))
        articles.append({"text": text, "published": pub, "source": "synthetic_news"})
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# 3.  REDDIT DATA  (PRAW with anonymous read-only access)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reddit_posts(
    subreddits: list[str] = REDDIT_SUBREDDITS,
    limit: int = 100,
) -> list[dict]:
    """
    Pull hot/new posts from financial subreddits using PRAW in read-only mode.
    Uses Reddit's public JSON endpoint as a fallback if PRAW auth fails.

    Returns list of dicts: {"text": str, "published": datetime, "source": "reddit"}.
    """
    posts = []

    # ── Attempt 1: PRAW with anonymous client ────────────────────────────────
    try:
        import praw
        reddit = praw.Reddit(
            client_id     = "market_sentiment_bot",   # placeholder - PRAW allows anonymous reads
            client_secret = "market_sentiment_secret",
            user_agent    = "MarketSentimentBot/1.0 by u/research_bot",
        )
        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.hot(limit=limit // len(subreddits)):
                    text = f"{post.title}. {post.selftext[:200]}"
                    pub  = datetime.datetime.fromtimestamp(post.created_utc)
                    posts.append({"text": text, "published": pub, "source": "reddit"})
            except Exception as sub_exc:
                logger.warning(f"PRAW subreddit '{sub_name}' failed: {sub_exc}")

        if posts:
            logger.info(f"  [OK] Fetched {len(posts)} Reddit posts via PRAW")
            return posts

    except Exception as exc:
        logger.warning(f"PRAW failed ({exc}). Trying public JSON API ...")

    # ── Attempt 2: Reddit public JSON (no auth required) ────────────────────
    headers = {"User-Agent": "MarketSentimentBot/1.0"}
    for sub_name in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub_name}/hot.json?limit=25"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for item in data["data"]["children"]:
                d = item["data"]
                text = f"{d.get('title', '')}. {d.get('selftext', '')[:200]}"
                pub  = datetime.datetime.fromtimestamp(d.get("created_utc", time.time()))
                posts.append({"text": text, "published": pub, "source": "reddit_json"})
            logger.info(f"  [OK] /r/{sub_name}: {len(posts)} posts via public JSON")
        except Exception as exc2:
            logger.warning(f"  Public JSON for /r/{sub_name} failed: {exc2}")

    if posts:
        return posts

    # ── Fallback: synthetic reddit posts ─────────────────────────────────────
    logger.warning("All Reddit sources failed - generating synthetic posts")
    return _generate_synthetic_reddit(limit)


def _generate_synthetic_reddit(n: int = 100) -> list[dict]:
    """Generate realistic WSB / investing-style posts."""
    templates = [
        "{ticker} to the moon!!! - why I'm holding 1000 shares",
        "Just bought {ticker} calls, {sentiment} about next earnings",
        "{ticker} is {evaluation} right now - change my mind",
        "DD on {ticker}: {analysis}",
        "Is {ticker} a buy at current prices? {reasoning}",
        "{ticker} short squeeze incoming? {evidence}",
        "Why I think {ticker} will {direction} by end of quarter",
        "Lost $10k on {ticker} puts. Here's what I learned.",
        "{ticker} earnings play: {position}. Any thoughts?",
        "Fundamental analysis of {ticker} - long-term buy",
    ]
    sentiments  = ["bullish", "bearish", "cautiously optimistic", "very confident", "nervous"]
    evaluations = ["undervalued", "overvalued", "fairly priced", "a steal", "a trap"]
    analyses    = ["strong revenue growth supports a higher valuation",
                   "margins are compressing but market ignores it",
                   "institutional buying is a green flag",
                   "short interest is rising - potential squeeze"]
    reasonings  = ["P/E ratio looks cheap vs sector", "momentum indicators look bullish",
                   "management has a proven track record", "competition is overstated"]
    evidences   = ["Short float is 25% - could be explosive",
                   "Options chain shows heavy call buying",
                   "Insider buying last week"]
    directions  = ["pop 20%", "dump hard", "consolidate then break out", "double"]
    positions   = ["10 calls at the money", "100 shares long", "put spread for downside hedge"]

    posts = []
    now   = datetime.datetime.now()
    for _ in range(n):
        tmpl   = random.choice(templates)
        ticker = random.choice(TICKERS)
        text   = (
            tmpl
            .replace("{ticker}",     ticker)
            .replace("{sentiment}",  random.choice(sentiments))
            .replace("{evaluation}", random.choice(evaluations))
            .replace("{analysis}",   random.choice(analyses))
            .replace("{reasoning}",  random.choice(reasonings))
            .replace("{evidence}",   random.choice(evidences))
            .replace("{direction}",  random.choice(directions))
            .replace("{position}",   random.choice(positions))
        )
        pub = now - datetime.timedelta(minutes=random.randint(0, 60 * 24 * 14))
        posts.append({"text": text, "published": pub, "source": "synthetic_reddit"})
    return posts


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TWITTER-LIKE DATA  (always synthetic – free-tier X/Twitter API is too limited)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_tweets(n: int = 200) -> list[dict]:
    """
    Generate realistic synthetic tweets about the tracked stocks.
    Twitter's API restrictions make free real-time collection impractical,
    so we produce statistically representative synthetic data.

    Returns list of dicts: {"text": str, "published": datetime, "source": "twitter"}.
    """
    logger.info("Generating synthetic tweet data ...")

    templates = [
        "${ticker} looking {adj} today. {action}",
        "Just {trade} ${ticker}. {reason}",
        "${ticker} earnings tomorrow - {position}",
        "Can't believe ${ticker} is {doing}. {reaction}",
        "${ticker} {chart_pattern}. {interpretation} #stocks #trading",
        "Watching ${ticker} closely. {observation}",
        "{feeling} on ${ticker} right now. {sentiment_stmt}",
        "${ticker} news: {headline_snippet}",
        "Hot take: ${ticker} is going to {prediction} by year end",
        "@trader ${ticker} {agreement}. {follow_up}",
    ]
    adjs            = ["strong", "weak", "volatile", "bullish", "bearish", "oversold", "overbought"]
    actions         = ["Loading up!", "Taking profits.", "Waiting for dip.", "YOLO calls!",
                       "Trimming position.", "Holding firm.", "Adding more."]
    trades          = ["bought", "sold", "shorted", "covered", "averaged down on"]
    reasons         = ["fundamentals are solid", "chart looks great", "earnings beat incoming",
                       "stop loss hit", "profit target reached", "technical breakout"]
    positions       = ["long calls", "long puts", "short", "long shares", "neutral"]
    doings          = ["dumping", "surging", "consolidating", "breaking out", "breaking down"]
    reactions       = ["Insane!", "Expected it.", "Buying the dip.", "Ouch.", "Let's go!!!"]
    patterns        = ["cup and handle forming", "double top", "bullish divergence",
                       "death cross", "golden cross", "ascending triangle"]
    interpretations = ["Bullish signal!", "Bearish reversal.", "Watch closely.",
                       "Big move incoming.", "Breakout soon."]
    observations    = ["Volume is picking up.", "Institutional buying detected.",
                       "Short interest rising.", "Options flow is bullish.", "Low volatility."]
    feelings        = ["Bullish", "Bearish", "Neutral", "Uncertain", "Very confident"]
    sentiment_stmts = ["Strong buy.", "Avoid for now.", "Hold and wait.",
                       "Risk/reward favors longs.", "Too risky here."]
    headlines       = ["beats earnings estimates", "misses revenue target",
                       "announces buyback program", "raises guidance",
                       "faces regulatory scrutiny", "expands into new market"]
    predictions     = ["hit ATH", "drop 15%", "double", "underperform", "surprise everyone"]
    agreements      = ["couldn't agree more", "disagree completely", "interesting take",
                       "spot on analysis", "missing key context"]
    follow_ups      = ["DM me your thoughts.", "What's your PT?", "DYOR.",
                       "NFA but interesting.", "Let's discuss."]

    tweets = []
    now    = datetime.datetime.now()
    for _ in range(n):
        tmpl   = random.choice(templates)
        ticker = random.choice(TICKERS)
        text   = (
            tmpl
            .replace("{ticker}",           ticker)
            .replace("{adj}",              random.choice(adjs))
            .replace("{action}",           random.choice(actions))
            .replace("{trade}",            random.choice(trades))
            .replace("{reason}",           random.choice(reasons))
            .replace("{position}",         random.choice(positions))
            .replace("{doing}",            random.choice(doings))
            .replace("{reaction}",         random.choice(reactions))
            .replace("{chart_pattern}",    random.choice(patterns))
            .replace("{interpretation}",   random.choice(interpretations))
            .replace("{observation}",      random.choice(observations))
            .replace("{feeling}",          random.choice(feelings))
            .replace("{sentiment_stmt}",   random.choice(sentiment_stmts))
            .replace("{headline_snippet}", random.choice(headlines))
            .replace("{prediction}",       random.choice(predictions))
            .replace("{agreement}",        random.choice(agreements))
            .replace("{follow_up}",        random.choice(follow_ups))
        )
        pub = now - datetime.timedelta(minutes=random.randint(0, 60 * 24 * 7))
        tweets.append({"text": text, "published": pub, "source": "synthetic_twitter"})

    logger.info(f"  [OK] Generated {len(tweets)} synthetic tweets")
    return tweets


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINED COLLECTOR  (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_data() -> tuple[dict, list[dict]]:
    """
    Master collection function called by the pipeline.

    Returns
    -------
    price_data : dict[ticker -> DataFrame]
    text_data  : list of {"text", "published", "source"} dicts
    """
    logger.info("=" * 60)
    logger.info("STEP 1 - DATA COLLECTION")
    logger.info("=" * 60)

    price_data = fetch_price_data()

    news   = fetch_news_headlines(max_articles=300)
    reddit = fetch_reddit_posts(limit=150)
    tweets = fetch_tweets(n=300)

    text_data = news + reddit + tweets
    logger.info(f"\nTotal text items collected: {len(text_data)}")
    logger.info(f"  News     : {len(news)}")
    logger.info(f"  Reddit   : {len(reddit)}")
    logger.info(f"  Tweets   : {len(tweets)}")

    return price_data, text_data


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prices, texts = collect_all_data()
    for t, df in prices.items():
        print(f"{t}: {len(df)} rows | last close = {df['Close'].iloc[-1]:.2f}")
    print(f"\nTotal text documents: {len(texts)}")
    print("Sample:", texts[0]["text"][:120])
