# 🥇 Gold Price Prediction Using Arabic Sentiment Analysis & CNN-LSTM

A multi-stage data science project that predicts gold prices by combining Arabic Twitter sentiment analysis with historical financial data, using a CNN-LSTM deep learning model. The core research question is: **does adding public sentiment from Arabic tweets improve gold price prediction accuracy?**

----

## 📌 Project Overview

The project is structured into three connected stages:

1. **Arabic Sentiment Analysis** — classify tweets about gold as Positive or Negative
2. **Historical Financial Data Integration** — merge gold prices with macroeconomic indicators
3. **CNN-LSTM Prediction** — predict gold prices with and without tweet sentiment, then compare results

---

## 📊 Datasets

### Twitter Data
- **Source:** Arabic tweets about gold (2021–2024), scraped and labeled
- **Labels:** 0 = Positive, 1 = Negative
- **Language:** Arabic

### Financial Data (`model_data-2021-2024.csv`)
- **Period:** March 2021 – March 2024 (769 daily records)
- **Features:** Gold price, Inflation rate, Exchange rate (SAR/USD), Petrochemical index, Oil price, Twitter dominant sentiment score

---

## 🔍 Project Pipeline

### Stage 1 — Arabic Sentiment Analysis

**Text Preprocessing (Arabic NLP)**
- Removed retweet tags, URLs, emails, emojis, and punctuation (Arabic + Latin)
- Stripped diacritics (Tashkeel) and Tatweel
- Normalized Hamza, Alef, and Lam-Alef forms
- Converted Eastern Arabic numerals to Western
- Applied custom Arabic stopword list (with domain-specific additions for financial terms)
- Tokenized using `pyarabic` and stemmed using NLTK's ISRIStemmer

**Text Vectorization**
- TF-IDF vectorization on stemmed Arabic text

**Models Trained & Compared**

| Model | Test Accuracy | Test F1-Score |
|---|---|---|
| Multinomial Naive Bayes | 75% | 0.77 |
| SVM (Linear kernel) | 87% | 0.89 |
| Soft Voting (NB + SVM) | **90%** | **0.91** |

- **Voting Classifier** (soft voting combining NB and SVM) achieved the best performance
- Learning curves plotted for both Naive Bayes and Voting Classifier
- Word clouds generated showing most predictive positive and negative Arabic words per model

**Sentiment Aggregation**
- Daily dominant sentiment score derived from tweet majority label per day
- Exported as `tweet_score.csv` for use in Stage 3

---

### Stage 2 — Historical Data Integration

- Merged 5 financial datasets by date: Gold, Inflation, Exchange Rate, Petrochemical, Oil
- Visualized missing data patterns using heatmaps and trend lines
- Filled missing values using forward-fill (`ffill`)
- Correlation matrix computed across all features
- Lag correlation analysis (up to 10 lags) to find the best historical lag for each variable relative to gold price
- Final merged dataset exported as `model_data-2021-2024.csv`

---

### Stage 3 — CNN-LSTM Gold Price Prediction

**Architecture**
- TimeDistributed Conv1D (50 filters, kernel=3, tanh activation)
- TimeDistributed MaxPooling1D
- TimeDistributed Flatten
- LSTM (100 units, tanh activation)
- Dropout (0.3)
- Dense (100 units, ReLU) → Dense (1 unit output)
- Optimizer: Adam (lr=0.01), Loss: MSE
- Sequence length: 10 timesteps
- Split: 70% train / 10% validation / 20% test

**Two Models Compared**

| Model | MAPE | RMSE |
|---|---|---|
| CNN-LSTM **without** sentiment | 2.65% | 1.28 |
| CNN-LSTM **with** sentiment | **1.38%** | **0.81** |

Adding the Arabic tweet sentiment score reduced MAPE by **~48%** and RMSE by **~37%**, confirming that public sentiment meaningfully improves gold price prediction.

**Evaluation includes:** Actual vs. Predicted plots, residual analysis, directional confusion matrix (rise/fall), and per-day metrics table (MSE, MAE, RMSE, MAPE).

---

## 📁 Project Structure

```
gold-sentiment-prediction/
│
├── sentiment_analysis.ipynb          # Full pipeline notebook
├── model_data-2021-2024.csv          # Final merged dataset (financial + sentiment)
└── README.md
```

> **Note:** Additional raw files used in the notebook (tweets CSV, Gold/Oil/Inflation CSVs, Arabic font for word clouds) are not included in this repo due to size. The final merged dataset above is sufficient to run Stage 3.

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/gold-sentiment-prediction.git
```

2. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras nltk pyarabic wordcloud imbalanced-learn tqdm
```

3. Download required NLTK data (run once)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords').
```

4. Open and run the notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

> **Important:** Update the file paths in the notebook from `/content/...` (Colab format) to relative paths matching your local folder structure.

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualizations |
| `nltk` / `pyarabic` | Arabic NLP (tokenization, stemming, stopwords) |
| `scikit-learn` | TF-IDF, SVM, Naive Bayes, Voting Classifier, metrics |
| `wordcloud` | Word cloud generation |
| `tensorflow` / `keras` | CNN-LSTM deep learning model |
| `tqdm` | Progress bars |

---

## 💡 Key Findings

- The **Soft Voting Classifier** (NB + SVM) achieved 90% accuracy and 0.91 F1-score on Arabic tweet sentiment classification
- Arabic public sentiment on Twitter is a **meaningful signal** for gold price movement — adding it reduced prediction error by nearly half
- The **CNN-LSTM with sentiment** reached a MAPE of 1.38% vs. 2.65% without it, a strong result for a daily financial time-series task
- Exchange rate and oil price showed the strongest lag correlations with gold price

---

## 👤 Author

**Khaled Abdulaziz**
