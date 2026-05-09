# Project 1: Real-Time Market Movement Prediction System

This project is an end-to-end financial machine learning pipeline that predicts stock market movements using a combination of price data and sentiment analysis.

## Features
- **Data Collection**: Fetches real-time price data (yfinance), news headlines (Reuters RSS), and social media sentiment (Reddit & Synthetic Twitter).
- **Sentiment Analysis**: Uses VADER to score documents and aggregate sentiment per ticker.
- **Deep Learning Models**: Implements RNN, LSTM, and GRU models for dual-task prediction (binary direction and return magnitude).
- **Automated Pipeline**: End-to-end execution from data ingestion to model evaluation and visualization.

## Structure
- `data_collector.py`: Scrapers and data generators.
- `sentiment_analyzer.py`: VADER-based sentiment processing.
- `dataset_builder.py`: Data merging and PyTorch dataset creation.
- `models.py`: RNN/LSTM/GRU architecture definitions.
- `trainer.py`: Training logic and combined loss optimization.
- `evaluator.py`: Performance metrics and plotting.
- `main.py`: Entry point.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the pipeline: `python main.py`

## Outputs
- `checkpoints/`: Saved model states.
- `data/`: Processed datasets in CSV format.
- `results/`: Comparison metrics in CSV format.
- `plots/`: Performance visualizations.



`IEEEtran` document class.

<details>
<summary>Click to expand LaTeX Report Code</summary>

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Real-Time Market Movement Prediction System}

\author{\IEEEauthorblockN{1\textsuperscript{st} Author Name}
\IEEEauthorblockA{\textit{Department} \\
\textit{University / Organization}\\
City, Country \\
email address}
}

\maketitle

\begin{abstract}
This paper presents an end-to-end financial machine learning pipeline that predicts stock market movements utilizing a combination of price data and sentiment analysis.
\end{abstract}

\begin{IEEEkeywords}
Machine Learning, Sentiment Analysis, Stock Market Prediction, Deep Learning, RNN, LSTM, GRU
\end{IEEEkeywords}

\section{Introduction}
Predicting stock market movements is a challenging task due to the high volatility and noisy nature of financial time-series data. This project aims to build an end-to-end pipeline that leverages both historical price data and real-time sentiment analysis from news and social media to forecast stock directions and return magnitudes. By integrating natural language processing with deep learning sequence models, we attempt to capture market sentiment alongside technical indicators.

\section{Methodology}
The methodology involves a dual-task prediction model. We collect real-time price data (via yfinance), news headlines (Reuters RSS), and social media sentiment (Reddit and synthetic Twitter data). Sentiment analysis is performed using VADER to score and aggregate sentiment per ticker. The data is merged and normalized, then passed through sequence models (RNN, LSTM, GRU) with a dual-head architecture: a classification head for predicting the binary next-day direction (up/down) and a regression head for predicting the next-day return magnitude.

\section{Dataset Overview}
The dataset comprises technical price features (Close, Volume, MA5, MA20, Daily Return, Volatility) and sentiment features (mean compound, sentiment sum, positive ratio, negative ratio, document count). The sequences are generated using a sliding window of length 10. The dataset is split chronologically into 70\% training, 15\% validation, and 15\% testing to prevent look-ahead data leakage, and is normalized using standard scaling based on the training set.

\section{Model Architecture Diagrams}
All three evaluated models (Vanilla RNN, LSTM, GRU) share a common sequence-processing base with a configurable hidden size (default 64) and number of layers (default 2), utilizing dropout and layer normalization. 
\begin{itemize}
    \item \textbf{Recurrent Core:} RNN, LSTM, or GRU layers process the input sequence.
    \item \textbf{Classification Head:} A linear layer followed by ReLU, Dropout, and a final linear layer outputting 2 logits for CrossEntropyLoss.
    \item \textbf{Regression Head:} A similar sequential linear block outputting a single scalar for return magnitude prediction.
\end{itemize}

\section{Results Comparison}
The models were evaluated on the AAPL, TSLA, GOOGL, MSFT, and AMZN tickers. The evaluation metrics included Accuracy, Precision, Recall, F1-Score, and Root Mean Square Error (RMSE).
\begin{table}[htbp]
\caption{Model Comparison Results (Excerpt)}
\begin{center}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
\textbf{Ticker} & \textbf{Model} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{RMSE} \\
\hline
AAPL & RNN & 0.8333 & 0.8333 & 1.0 & 0.9091 & 0.181 \\
AAPL & LSTM & 0.3333 & 0.6667 & 0.4 & 0.5000 & 0.093 \\
AAPL & GRU & 0.1667 & 0.0 & 0.0 & 0.0 & 0.095 \\
\hline
TSLA & RNN & 0.8333 & 0.8333 & 1.0 & 0.9091 & 0.303 \\
TSLA & LSTM & 0.5000 & 1.0000 & 0.4 & 0.5714 & 0.074 \\
TSLA & GRU & 0.1667 & 0.0 & 0.0 & 0.0 & 0.097 \\
\hline
\end{tabular}
\label{tab1}
\end{center}
\end{table}

\section{Challenges Faced}
Key challenges encountered during the project included:
\begin{itemize}
    \item Aligning different frequencies of data (trading days vs. continuous news/social media sentiment).
    \item Handling missing sentiment data through forward-filling and imputation.
    \item Mitigating vanishing and exploding gradients in the Vanilla RNN, requiring gradient clipping.
    \item Preventing data leakage by strictly maintaining a chronological split for training, validation, and testing sets.
\end{itemize}

\section{Conclusion}
The project successfully developed a pipeline combining financial time-series and natural language sentiment to predict market movement. While the RNN model showed high classification metrics on some tickers, the LSTM and GRU models generally demonstrated lower RMSE, indicating better performance on the regression task and greater stability. Future work could explore transformer-based architectures and more sophisticated language models like FinBERT.

\section{References}
\begin{thebibliography}{00}
\bibitem{b1} C. J. Hutto and E. Gilbert, ``VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text,'' in \textit{Eighth International AAAI Conference on Weblogs and Social Media}, 2014.
\bibitem{b2} S. Hochreiter and J. Schmidhuber, ``Long Short-Term Memory,'' \textit{Neural Computation}, vol. 9, no. 8, pp. 1735--1780, 1997.
\bibitem{b3} K. Cho et al., ``Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation,'' in \textit{EMNLP}, 2014.
\end{thebibliography}

\end{document}
```
</details>
