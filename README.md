# King County House Price Prediction

> Predicting house sale prices in King County (WA) and identifying what makes a home **"luxury"** — an end‑to‑end Machine Learning project covering regression, classification and clustering.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Ironhack-Data%20Analytics-2EE6D6">
</p>

---

## Project Overview

This project analyzes house sales in **King County** (which includes Seattle) for transactions between **May 2014 and May 2015**. The dataset contains **21,613 properties** described by **21 attributes** (size, quality, condition, location, renovation status, etc.).

The work pursues three complementary objectives:

1. **Regression** — Predict the continuous sale `price` of a house and understand which features drive it.
2. **Classification** — Predict whether a house belongs to the **luxury segment** (sale price **≥ $650,000**).
3. **Clustering** — Discover natural groups of houses based purely on their physical and location characteristics, using unsupervised learning.

**Author:** Daniel Fernández

---

## Learning Objectives

- Build a **complete, leakage‑free ML pipeline** from raw data to evaluated models.
- Apply **feature engineering** and **target transformation** to a strongly skewed target.
- Compare multiple **regression and classification** algorithms with consistent, reproducible evaluation.
- Use **unsupervised learning (KMeans)** to segment the housing market.
- Translate model output into **business‑readable insights**.

---

## Project Structure

```
.
├── Project.ipynb          # Main notebook (full analysis & models)
├── data/
│   └── king_country_houses_aa.csv
└── README.md
```

The notebook itself is organised in seven stages:

1. Data loading and initial exploration
2. Data cleaning and feature engineering
3. Exploratory Data Analysis (EDA) + hypothesis testing
4. Price prediction — regression models
5. Luxury‑home prediction — classification models
6. House segmentation — KMeans clustering
7. Conclusions and recommendations

---

##  Data Cleaning & Feature Engineering

| New feature | Description |
|---|---|
| `log_price` | `log1p(price)` — used to handle the right‑skewed target |
| `luxury_home` | Binary flag: `1` if price ≥ $650,000, else `0` |
| `sale_year`, `sale_month` | Extracted from the transaction `date` |
| `house_age` | `sale_year − yr_built` |
| `was_renovated` | `1` if the home has ever been renovated |
| `years_since_renovation` | Years since last renovation (`0` if never) |
| `has_basement` | `1` if `sqft_basement > 0` |
| `price_per_sqft` | `price / sqft_living` (analysis only) |
| `zipcode` | Cast to categorical (string) for one‑hot encoding |

###  Target Leakage Prevention
Variables **derived from price** (`log_price`, `luxury_home`, `price_per_sqft`) plus identifiers (`id`, `date`, `yr_built`, `yr_renovated`) are **excluded from the regression feature set**, so the model never "peeks" at the answer.

---

##  Key EDA Findings

- The price distribution is **strongly right‑skewed** (mean **$540,088**, median **$450,000**, max **$7.7M**), which motivates the log transformation.
- Strongest linear correlations with price: **`sqft_living` (0.70)**, **`grade` (0.67)**, `sqft_above` (0.61), `sqft_living15` (0.59), `bathrooms` (0.53).
- **Hypothesis test** (Welch's t‑test on `log_price`): homes with **grade ≥ 8** are significantly more expensive than lower‑grade homes — `t = 96.12`, `p ≈ 0`. Average price: **$714,685** (grade ≥ 8) vs **$380,564** (grade < 8).
- The luxury segment (≥ $650k) represents **24.6%** of the market and differs sharply on size, grade and view quality.

---

##  Modelling

All models share the same preprocessing inside a **scikit‑learn `Pipeline`**: `StandardScaler` for numeric features + `OneHotEncoder` for `zipcode`, with an **80/20 train‑test split** (`random_state=42`). Regressors are wrapped in a `TransformedTargetRegressor` (`log1p` / `expm1`) so metrics stay in dollars.

### 1️ Regression — predicting `price`

| Model | Test R² | Test MAE | Test RMSE | Overfit (Train−Test R²) |
|---|---|---|---|---|
| ** Gradient Boosting** | **0.88** | **$76,567** | **$137,272** | **0.01** |
| Random Forest | 0.87 | $73,142 | $139,056 | 0.10 |
| Decision Tree | 0.82 | $91,288 | $166,481 | 0.09 |
| KNN Regressor | 0.78 | $91,802 | $182,292 | 0.06 |
| Ridge | 0.71 | $80,356 | $208,020 | 0.12 |
| Linear Regression | 0.71 | $80,406 | $208,592 | 0.12 |
| Lasso | 0.69 | $89,222 | $217,192 | 0.15 |

> **Gradient Boosting** was selected as the best model: highest Test R², lowest Test RMSE, and almost **no overfitting** (Random Forest had a slightly better MAE but a much larger train/test gap).

**Top price drivers (grouped feature importance):** `lat` (0.31), `grade` (0.28), `sqft_living` (0.28) — i.e. **where the house is, how well it's built, and how big it is**.

**Where the model struggles:** error grows with price. Average absolute error is **~$48k for non‑luxury** homes but **~$158k for luxury** homes — expensive, rarer properties are harder to price.

###  Classification — predicting `luxury_home`

| Model | Test Accuracy | Precision (Luxury) | Recall (Luxury) | F1 (Luxury) |
|---|---|---|---|---|
| ** Gradient Boosting** | **0.93** | **0.89** | **0.80** | **0.84** |
| Random Forest | 0.92 | 0.90 | 0.78 | 0.84 |
| Decision Tree | 0.92 | 0.84 | 0.82 | 0.83 |
| KNN | 0.91 | 0.85 | 0.77 | 0.81 |

> The **Gradient Boosting Classifier** reaches **93% accuracy** and an **F1 of 0.84** on the luxury class. Most errors are **missed luxury homes** (218 false negatives vs 103 false positives) — the model is slightly conservative about labelling a home as luxury.

###  Clustering — KMeans segmentation

Using 18 structural/location features (scaled), the **silhouette score peaked at k = 2**, producing two clean market segments:

| Cluster | Homes | Avg price | Avg living area | Avg grade | Luxury rate | Profile |
|---|---|---|---|---|---|---|
| **Premium** | 8,210 | **$736,090** | 2,842 sqft | 8.64 | 46% | Larger, newer, higher‑grade homes |
| **Standard** | 13,403 | **$420,027** | 1,613 sqft | 7.06 | 12% | Smaller, older, mainstream homes |

---

##  Tools & Technologies

| Tool | Role |
|---|---|
| **Python 3.11** | Core language |
| **pandas / NumPy** | Data manipulation & feature engineering |
| **Matplotlib / Seaborn** | Visualization & EDA |
| **SciPy** | Hypothesis testing (`ttest_ind`) |
| **scikit‑learn** | Pipelines, preprocessing, regression, classification, KMeans |
| **Jupyter Notebook** | Interactive analysis environment |

---

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/king-county-house-price-prediction.git
cd king-county-house-price-prediction

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter

# 4. Place the dataset in ./data/  then launch the notebook
jupyter notebook Project.ipynb
```

> 📎 **Dataset note:** the notebook expects the CSV at `data/king_country_houses_aa.csv`. Update the `pd.read_csv(...)` path in the first code cell if your file lives elsewhere.

---

##  Conclusions

- **House prices in King County are driven mostly by location, build quality and size.** A well‑tuned **Gradient Boosting** model explains **~88%** of price variance.
- The same features that drive price also **cleanly separate luxury from non‑luxury homes** (93% classification accuracy).
- The market naturally splits into a **premium** and a **standard** segment — without ever using price to define them.
- The main limitation is **accuracy on very expensive homes**, where the model's error grows substantially. More data on the luxury tail (or a dedicated high‑end model) would be a sensible next step.

---

##  Possible Next Steps

- Hyperparameter tuning with cross‑validation (`GridSearchCV` / `RandomizedSearchCV`).
- A dedicated model for the luxury tail to reduce high‑end error.
- Geospatial enrichment (distance to Seattle, schools, waterfront proximity).
- Deploy the best model behind a simple price‑estimation API or app.

---

<p align="center"><b>ironhack</b> · Data Analytics Bootcamp · Final Project</p>
