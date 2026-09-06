# Brian | Data & AI Portfolio — Streamlit App

An interactive portfolio web app built with **Streamlit**, showcasing exploratory data analysis, a trained machine-learning demand forecasting model, and a property price estimator — all in one deployable app.

Built by **Brianca Hernawan, BBus, MBA** — Head of AIML, Data Science & Data Analytics, BBF Meat.

---

## ✨ Features

The app is organised into four tabs:

### 🏠 Home
Professional profile, headline metrics (experience, industries, clients, cost savings) and a featured-project gallery.

### 📊 EDA Dashboard
Interactive exploratory data analysis on the classic **Iris** dataset (loaded from `scikit-learn`):

- Multi-select species filter + dynamic X/Y axis selection
- Live summary metrics (samples, features, missing values)
- Bar chart of feature means by species
- Donut chart of class distribution
- Box plots with all data points overlaid
- Histogram with violin (KDE-style) marginal
- Scatter plot with OLS trendline (`statsmodels`)
- Toggleable raw-data table

### 📈 Demand Forecasting
Product demand prediction powered by a trained **XGBoost regressor** (`xgboost_demand_model.pkl`) with categorical features handled by saved `LabelEncoder` objects (`label_encoders.pkl`).

- **Manual Input** — single-record prediction from Price, Discount, Inventory Level, Promotion, Competitor Pricing and Category
- **Batch Prediction** — upload a CSV, run the full pipeline, preview results and download predictions as CSV
- Column validation with clear error messaging on schema mismatch
- Feature-importance insight: Promotion (58%) and Category (27%) dominate the model

Supported categories: `Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`.

### 🏠 ML Predictor — Jakarta House Prices
Property price estimator derived from a model trained on **10,000+ Jakarta listings** scraped from rumahrumah.com (R² = 92%, MAPE = 11%). Inputs land area, building area, bedrooms, bathrooms and certificate status (SHM vs non-SHM), then returns an estimated price in IDR with a component-level breakdown chart.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| Data handling | pandas, NumPy |
| Visualisation | Plotly Express, Matplotlib, Seaborn |
| ML / Stats | XGBoost, scikit-learn, statsmodels |

---

## 📁 Repository Structure

```
.
├── app.py                        # Main Streamlit application
├── xgboost_demand_model.pkl      # Trained XGBoost demand model
├── label_encoders.pkl            # Fitted LabelEncoders for categorical features
├── Brain_pasphoto.jpeg           # Profile photo (Home tab)
├── requirements.txt              # Python dependencies
└── README.md
```

> All `.pkl` files and the profile photo must sit in the **same folder as `app.py`** — the app loads them by relative path.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📂 Batch Prediction — CSV Format

The batch uploader expects these exact column headers:

| Column | Type | Example |
|---|---|---|
| `Price` | float | 49.99 |
| `Discount` | int | 10 |
| `Inventory Level` | int | 120 |
| `Promotion` | int (0/1) | 1 |
| `Competitor Pricing` | float | 52.50 |
| `Category` | string | Electronics |

Sample row:

```csv
Price,Discount,Inventory Level,Promotion,Competitor Pricing,Category
49.99,10,120,1,52.50,Electronics
```

Unknown category values will raise an encoder error — use only the five supported categories listed above.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select the repo, branch, and set the main file to `app.py`.
4. Deploy. Dependencies install automatically from `requirements.txt`.

**Tip:** pin versions in `requirements.txt` (e.g. `scikit-learn==1.6.1`) so the deployed environment matches the one that produced the `.pkl` files — the encoders were fitted with scikit-learn 1.6.1 and a mismatched version can throw unpickling warnings or errors.

---

## 🗺️ Roadmap

- [ ] Swap the house-price tab to load the trained model artefact directly
- [ ] Add SHAP explainability to the demand forecast output
- [ ] Time-series demand view with seasonality decomposition
- [ ] Downloadable EDA report (PDF)

---

## 📬 Contact

**Brianca Hernawan** — Head of AIML, Data Science & Data Analytics, BBF Meat
MBA, IPMI International Business School · BBus, Victoria University

- LinkedIn: `<https://www.linkedin.com/in/brian-hernawan/>`
- Email: `brian.hernawan@gmail.com`

---

## 📄 License

MIT — feel free to fork and adapt.
