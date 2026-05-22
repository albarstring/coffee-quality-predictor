import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
import streamlit as st
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(
    page_title="☕ Coffee Quality Predictor",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_PATH = Path(__file__).with_name("coffee_quality_artifacts_v3.joblib")

st.markdown("""
<style>
* { font-family: 'Segoe UI', Tahoma, sans-serif; }
body { background: #f5f3f0; }
.stApp { background: #f5f3f0; }

/* Main content */
.main-header { 
    font-size: 2.5rem; 
    font-weight: 800; 
    color: #3d2b1f; 
    margin-bottom: 0.5rem;
}
.main-subtitle { 
    font-size: 1rem; 
    color: #666; 
    margin-bottom: 1.5rem;
}

/* Cards */
.info-card {
    background: white;
    border-left: 4px solid #8B6F47;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.prediction-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #4a3728 0%, #6b5344 100%);
}
div[data-testid="stSidebar"] * {
    color: #f5f3f0 !important;
}
div[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffd9a8 !important;
    font-size: 1.1rem !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.8rem !important;
}
div[data-testid="stSidebar"] .stSlider label,
div[data-testid="stSidebar"] .stSelectbox label {
    color: #f5f3f0 !important;
    font-weight: 600;
}

/* Results */
.result-premium { color: #4CAF50; }
.result-specialty { color: #FF9800; }
.result-good { color: #2196F3; }
.result-below { color: #F44336; }

.score-big { 
    font-size: 3rem; 
    font-weight: 800;
    margin: 10px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: white;
    border-bottom: 2px solid #ddd;
}
.stTabs [role="tablist"] button {
    color: #666;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
}
.stTabs [role="tab"][aria-selected="true"] {
    color: #8B6F47;
    border-bottom: 3px solid #8B6F47;
}

/* Table */
.category-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 8px;
    overflow: hidden;
}
.category-table th {
    background: #8B6F47;
    color: white;
    padding: 12px;
    text-align: left;
    font-weight: 700;
}
.category-table td {
    padding: 12px;
    border-bottom: 1px solid #eee;
    color: #3d2b1f;
}
.category-table tr:last-child td {
    border-bottom: none;
}

/* Metrics */
.metric-box {
    background: white;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.metric-label { 
    font-size: 0.85rem; 
    color: #999; 
    margin-bottom: 5px;
}
.metric-value { 
    font-size: 1.8rem; 
    font-weight: 800; 
    color: #3d2b1f;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="☕ Loading model and data...")
def load_and_train():
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)

    def random_oversample(X_data, y_data, random_state=42):
        frame = X_data.copy()
        frame["__target__"] = y_data
        max_count = frame["__target__"].value_counts().max()
        balanced_parts = []
        for target_value, group in frame.groupby("__target__"):
            if len(group) < max_count:
                sampled = group.sample(max_count, replace=True, random_state=random_state)
            else:
                sampled = group
            balanced_parts.append(sampled)
        balanced = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
        y_balanced = balanced.pop("__target__")
        return balanced, y_balanced

    url_arabica = "https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv"
    url_robusta = "https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/robusta_data_cleaned.csv"

    df_arabica = pd.read_csv(url_arabica)
    df_robusta = pd.read_csv(url_robusta)
    df_arabica["Species"] = "Arabica"
    df_robusta["Species"] = "Robusta"

    kolom_bersama = list(set(df_arabica.columns) & set(df_robusta.columns))
    df = pd.concat([df_arabica[kolom_bersama], df_robusta[kolom_bersama]], ignore_index=True)

    sensori = [
        col for col in ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points"]
        if col in df.columns
    ]

    df = df.dropna(subset=["Total.Cup.Points"])
    df = df[df["Total.Cup.Points"] > 0]

    def kategori_kopi(score):
        if score >= 87:
            return "🏆 Specialty Premium"
        if score >= 80:
            return "⭐ Specialty"
        if score >= 70:
            return "✅ Good Quality"
        return "⚠️ Below Standard"

    df["Kualitas"] = df["Total.Cup.Points"].apply(kategori_kopi)

    df_model = df.copy()
    fitur_numerik = [col for col in sensori if col in df_model.columns]
    fitur_kat = [col for col in ["Species", "Processing.Method", "Color"] if col in df_model.columns]

    for col in fitur_numerik:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    for col in fitur_kat:
        df_model[col] = df_model[col].fillna("Unknown")

    le_dict = {}
    for col in fitur_kat:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    target_map = {
        "⚠️ Below Standard": 0,
        "✅ Good Quality": 1,
        "⭐ Specialty": 2,
        "🏆 Specialty Premium": 3,
    }
    df_model["target"] = df_model["Kualitas"].map(target_map)
    df_model = df_model.dropna(subset=["target"])
    df_model["target"] = df_model["target"].astype(int)

    semua_fitur = fitur_numerik + fitur_kat
    X = df_model[semua_fitur]
    y = df_model["target"]

    X, y = random_oversample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False, eval_metric="mlogloss"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hasil = {}
    for nama, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        y_pred = mdl.predict(X_test_s)
        hasil[nama] = {
            "model": mdl,
            "y_pred": y_pred,
            "acc": accuracy_score(y_test, y_pred) * 100,
            "f1": f1_score(y_test, y_pred, average="weighted") * 100,
            "cv": cross_val_score(mdl, X_train_s, y_train, cv=skf, scoring="accuracy").mean() * 100,
        }

    best_name = max(hasil, key=lambda x: hasil[x]["acc"])
    best_mdl = hasil[best_name]["model"]

    artifacts = {
        "df": df,
        "df_arabica": df_arabica,
        "df_robusta": df_robusta,
        "sensori": sensori,
        "semua_fitur": semua_fitur,
        "fitur_numerik": fitur_numerik,
        "fitur_kat": fitur_kat,
        "le_dict": le_dict,
        "scaler": scaler,
        "X_test_s": X_test_s,
        "y_test": y_test,
        "hasil": hasil,
        "best_name": best_name,
        "best_mdl": best_mdl,
        "target_map": target_map,
    }

    joblib.dump(artifacts, ARTIFACT_PATH, compress=3)
    return artifacts


try:
    data = load_and_train()
except Exception as exc:
    st.error(f"❌ Gagal memuat model: {exc}")
    st.stop()

df = data["df"]
df_arabica = data["df_arabica"]
df_robusta = data["df_robusta"]
sensori = data["sensori"]
semua_fitur = data["semua_fitur"]
fitur_numerik = data["fitur_numerik"]
fitur_kat = data["fitur_kat"]
le_dict = data["le_dict"]
scaler = data["scaler"]
X_test_s = data["X_test_s"]
y_test = data["y_test"]
hasil = data["hasil"]
best_name = data["best_name"]
best_mdl = data["best_mdl"]
target_map = data["target_map"]

label_names = ["Below Standard", "Good Quality", "Specialty", "Specialty Premium"]
label_emoji = {
    0: ("⚠️", "BELOW STANDARD", "result-below"),
    1: ("✅", "GOOD QUALITY", "result-good"),
    2: ("⭐", "SPECIALTY", "result-specialty"),
    3: ("🏆", "SPECIALTY PREMIUM", "result-premium"),
}


def prediksi(aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean_cup, sweetness, cupper, species, processing):
    sp_encoded = le_dict["Species"].transform([species])[0] if "Species" in le_dict else 0
    
    # Handle unseen processing method labels
    if "Processing.Method" in le_dict:
        valid_classes = le_dict["Processing.Method"].classes_
        if processing not in valid_classes:
            processing = valid_classes[0]  # Default to first known class
        proc_encoded = le_dict["Processing.Method"].transform([processing])[0]
    else:
        proc_encoded = 0

    input_row = {col: 0 for col in semua_fitur}
    sensor_values = {
        "Aroma": aroma,
        "Flavor": flavor,
        "Aftertaste": aftertaste,
        "Acidity": acidity,
        "Body": body,
        "Balance": balance,
        "Uniformity": uniformity,
        "Clean.Cup": clean_cup,
        "Sweetness": sweetness,
        "Cupper.Points": cupper,
    }
    for col, value in sensor_values.items():
        if col in input_row:
            input_row[col] = value

    if "Species" in input_row:
        input_row["Species"] = sp_encoded
    if "Processing.Method" in input_row:
        input_row["Processing.Method"] = proc_encoded
    if "Color" in input_row:
        input_row["Color"] = 0

    try:
        input_df = pd.DataFrame([input_row], columns=semua_fitur)
        input_s = scaler.transform(input_df)
        pred = best_mdl.predict(input_s)[0]
        proba = best_mdl.predict_proba(input_s)[0]
        return pred, proba
    except Exception as e:
        # Fallback: return neutral prediction
        return 1, [0.25, 0.25, 0.25, 0.25]


st.markdown('<p class="main-header">☕ Coffee Quality Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Real-time ensemble model untuk memprediksi kualitas kopi</p>', unsafe_allow_html=True)

st.sidebar.markdown("### 🫘 Coffee Specification")
species = st.sidebar.selectbox("🌿 Species", ["Arabica", "Robusta"], key="species")

# Get valid processing methods from trained LabelEncoder
valid_processing_methods = list(le_dict["Processing.Method"].classes_) if "Processing.Method" in le_dict else ["Washed / Wet", "Natural / Dry", "Pulped natural / honey"]
processing = st.sidebar.selectbox("⚙️ Processing Method", valid_processing_methods, key="processing")

st.sidebar.markdown("### 🎯 Sensory Attributes (0-10)")
aroma = st.sidebar.slider("👃 Aroma", 0.0, 10.0, 7.5, 0.5, key="aroma")
flavor = st.sidebar.slider("👅 Flavor", 0.0, 10.0, 7.5, 0.5, key="flavor")
aftertaste = st.sidebar.slider("🔄 Aftertaste", 0.0, 10.0, 7.2, 0.5, key="aftertaste")
acidity = st.sidebar.slider("🍋 Acidity", 0.0, 10.0, 7.3, 0.5, key="acidity")
body = st.sidebar.slider("💪 Body", 0.0, 10.0, 7.4, 0.5, key="body")
balance = st.sidebar.slider("⚖️ Balance", 0.0, 10.0, 7.3, 0.5, key="balance")
uniformity = st.sidebar.slider("🔁 Uniformity", 0.0, 10.0, 9.8, 0.5, key="uniformity")
clean_cup = st.sidebar.slider("🫙 Clean Cup", 0.0, 10.0, 9.8, 0.5, key="clean_cup")
sweetness = st.sidebar.slider("🍯 Sweetness", 0.0, 10.0, 9.8, 0.5, key="sweetness")
cupper = st.sidebar.slider("☕ Cupper Points", 0.0, 10.0, 7.5, 0.5, key="cupper")

# Real-time prediction
pred, proba = prediksi(aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean_cup, sweetness, cupper, species, processing)
em, lb, css_class = label_emoji[pred]
total_est = aroma + flavor + aftertaste + acidity + body + balance + uniformity + clean_cup + sweetness + cupper

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "🤖 Models"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**📋 Model Information**")
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-box"><div class="metric-label">Best Algorithm</div><div class="metric-value">{best_name}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value">{hasil[best_name]["acc"]:.1f}%</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('**📊 Quality Categories**')
        st.markdown("""
<table class="category-table">
  <tr>
    <th>Category</th>
    <th>Score</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>🏆 Specialty Premium</td>
    <td>≥ 87</td>
    <td>World-class coffee</td>
  </tr>
  <tr>
    <td>⭐ Specialty</td>
    <td>80–86</td>
    <td>Specialty grade</td>
  </tr>
  <tr>
    <td>✅ Good Quality</td>
    <td>70–79</td>
    <td>Good quality</td>
  </tr>
  <tr>
    <td>⚠️ Below Standard</td>
    <td>&lt; 70</td>
    <td>Needs improvement</td>
  </tr>
</table>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("**🏅 Quality Prediction**")
        st.markdown(f'<div class="score-big {css_class}">{em} {lb}</div>', unsafe_allow_html=True)
        st.markdown(f"**Estimated Cup Points:** {total_est:.1f} / 100")

        st.markdown("**Probability by Category:**")
        cats = ["🏆 Premium", "⭐ Specialty", "✅ Good", "⚠️ Below"]
        probs = [proba[3], proba[2], proba[1], proba[0]]
        colors_p = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]

        for cat, prob, clr in zip(cats, probs, colors_p):
            bar_w = int(prob * 100)
            st.markdown(f"""
<div style="margin-bottom: 12px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.9rem; color: #333;">
        <span>{cat}</span>
        <span style="font-weight: 700; color: {clr};">{prob*100:.1f}%</span>
    </div>
    <div style="background: #eee; border-radius: 6px; height: 8px; overflow: hidden;">
        <div style="background: {clr}; width: {bar_w}%; height: 100%; transition: width 0.3s;"></div>
    </div>
</div>
            """, unsafe_allow_html=True)

        reco_map = {
            3: "🏆 **Excellent!** Maintain quality control throughout processing.",
            2: "⭐ **Great!** Perfect for premium specialty coffee shops.",
            1: "✅ **Good** Improve balance & aftertaste for specialty grade.",
            0: "⚠️ **Needs Work** Focus on fermentation and drying process.",
        }
        st.info(reco_map[pred])
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("**📊 Dataset Overview**")
    filtered_df = df.copy()
    if "Species" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Species"] == species]
    if "Processing.Method" in filtered_df.columns and processing in filtered_df["Processing.Method"].astype(str).unique():
        filtered_df = filtered_df[filtered_df["Processing.Method"] == processing]
    if filtered_df.empty:
        filtered_df = df.copy()

    st.caption(f"Filtered view based on sidebar selection: {species} • {processing}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples", f"{len(filtered_df):,}")
    m2.metric("Arabica", f"{int((filtered_df['Species'] == 'Arabica').sum()):,}")
    m3.metric("Robusta", f"{int((filtered_df['Species'] == 'Robusta').sum()):,}")
    m4.metric("Avg Score", f"{filtered_df['Total.Cup.Points'].mean():.2f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Score Distribution**")
        fig = px.histogram(filtered_df, x="Total.Cup.Points", nbins=30, title="", labels={"Total.Cup.Points": "Cup Points", "count": "Count"})
        fig.add_vline(x=80, line_dash="dash", line_color="#FF9800", annotation_text="Specialty (80)", annotation_position="top right")
        fig.add_vline(x=87, line_dash="dash", line_color="#4CAF50", annotation_text="Premium (87)", annotation_position="top left")
        fig.update_traces(marker_color="#8B6F47")
        fig.update_layout(template="plotly_white", height=400, showlegend=False, paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Quality Distribution**")
        kualitas_count = filtered_df["Kualitas"].value_counts()
        colors_kopi = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        fig = go.Figure(data=[go.Pie(labels=kualitas_count.index, values=kualitas_count.values, marker=dict(colors=colors_kopi))])
        fig.update_layout(template="plotly_white", height=400, paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Sensory Correlation**")
        sensori_corr = filtered_df[sensori + ["Total.Cup.Points"]].corr()
        fig = px.imshow(sensori_corr, color_continuous_scale="YlOrBr", aspect="auto", labels=dict(color="Correlation"))
        fig.update_layout(template="plotly_white", height=450, paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Species Comparison**")
        species_quality = filtered_df.groupby(["Species", "Kualitas"]).size().unstack(fill_value=0)
        fig = go.Figure()
        colors_kopi = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        for i, quality in enumerate(species_quality.columns):
            fig.add_trace(go.Bar(x=species_quality.index, y=species_quality[quality], name=quality, marker_color=colors_kopi[i]))
        fig.update_layout(barmode="group", template="plotly_white", height=400, xaxis_title="Species", yaxis_title="Count", paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("**📈 Model Performance**")
    nama_list = list(hasil.keys())
    df_eval = pd.DataFrame({
        "Model": nama_list,
        "Accuracy": [hasil[n]["acc"] for n in nama_list],
        "F1-Score": [hasil[n]["f1"] for n in nama_list],
        "CV Score": [hasil[n]["cv"] for n in nama_list],
    }).set_index("Model")
    st.dataframe(df_eval.style.format("{:.2f}").highlight_max(color="#d4edda", axis=0), use_container_width=True)

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_test, hasil[best_name]["y_pred"])
        fig = px.imshow(cm, x=label_names, y=label_names, color_continuous_scale="YlOrBr", aspect="auto", text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(template="plotly_white", height=450, paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"))
        st.plotly_chart(fig, use_container_width=True)

    with col_fi:
        if hasattr(best_mdl, "feature_importances_"):
            st.markdown("**Feature Importance**")
            fi = pd.Series(best_mdl.feature_importances_, index=semua_fitur).sort_values(ascending=True)
            fig = px.bar(x=fi.values, y=fi.index, orientation="h", labels={"x": "Importance", "y": "Feature"}, title="")
            fig.update_traces(marker_color="#8B6F47")
            fig.update_layout(template="plotly_white", height=450, paper_bgcolor="#f5f3f0", plot_bgcolor="#f5f3f0", font=dict(color="#3d2b1f"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem;'>☕ Coffee Quality Predictor • Dataset: CQI • Model: Ensemble Learning</div>", unsafe_allow_html=True)
