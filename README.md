# ☕ Coffee Quality Predictor

Prediksi kualitas kopi Arabika & Robusta menggunakan **Ensemble Learning** berdasarkan karakteristik sensori.

## 🚀 Features
- **Real-time Prediction** — Hasil instan saat mengubah slider
- **Interactive Charts** — Visualisasi dinamis dengan Plotly
- **4 Model Ensemble** — Random Forest, XGBoost, Gradient Boosting, Logistic Regression
- **Dataset CQI** — 1000+ sampel kopi dari 30+ negara

## 📊 Fitur Prediksi
- 🌿 Species: Arabica / Robusta
- ⚙️ Processing Method: Washed, Natural, Pulped Natural, Semi-Washed
- 👃 10 Sensory Attributes: Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, Cupper Points

## 📈 Akurasi
- **Best Model**: Random Forest / XGBoost
- **Accuracy**: ~87-88%
- **F1-Score**: ~87%

## 🛠️ Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app.py
```

Buka http://localhost:8501

## 📦 Dataset
- Arabica: https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv
- Robusta: https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/robusta_data_cleaned.csv

## 🌐 Deployment
Deployed on **Streamlit Cloud** (Free)

---
**Created by:** Data Mining UTS Project | **Dataset:** Coffee Quality Institute (CQI)
