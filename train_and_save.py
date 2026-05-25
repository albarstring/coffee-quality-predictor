import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


ARTIFACT_PATH = Path(__file__).with_name("coffee_quality_artifacts_v3.joblib")


def smote_resample(X_data, y_data, random_state=42, k_neighbors=5):
    rng = np.random.default_rng(random_state)
    X_frame = pd.DataFrame(X_data).reset_index(drop=True)
    y_series = pd.Series(y_data).reset_index(drop=True)

    class_counts = y_series.value_counts().to_dict()
    max_count = max(class_counts.values())

    combined_frames = [X_frame.assign(__target__=y_series)]

    for class_label, count in class_counts.items():
        if count >= max_count:
            continue

        class_mask = y_series == class_label
        X_class = X_frame.loc[class_mask].reset_index(drop=True)
        n_samples_needed = max_count - count

        if len(X_class) == 1:
            synthetic_samples = np.repeat(X_class.values, n_samples_needed, axis=0)
        else:
            n_neighbors = min(k_neighbors, len(X_class) - 1)
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(X_class)
            neighbor_indices = nn.kneighbors(X_class, return_distance=False)

            synthetic_samples = []
            for _ in range(n_samples_needed):
                sample_idx = rng.integers(0, len(X_class))
                sample = X_class.iloc[sample_idx].to_numpy(dtype=float)

                neighbors = neighbor_indices[sample_idx][1:]
                neighbor_idx = rng.choice(neighbors)
                neighbor = X_class.iloc[neighbor_idx].to_numpy(dtype=float)

                gap = rng.random()
                synthetic = sample + gap * (neighbor - sample)
                synthetic_samples.append(synthetic)

            synthetic_samples = np.asarray(synthetic_samples)

        synthetic_frame = pd.DataFrame(synthetic_samples, columns=X_frame.columns)
        synthetic_targets = pd.Series([class_label] * len(synthetic_frame))

        combined_frames.append(synthetic_frame.assign(__target__=synthetic_targets))

    balanced = pd.concat(combined_frames, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_balanced = balanced.pop("__target__")
    return balanced, y_balanced


def train_and_save():
    print("Starting training. This may take several minutes...")
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

    X, y = smote_resample(X, y)

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

    best_name = max(hasil, key=lambda x: hasil[x]["acc"]) if hasil else None
    best_mdl = hasil[best_name]["model"] if best_name else None

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
    print(f"Saved artifacts to {ARTIFACT_PATH}")


if __name__ == '__main__':
    train_and_save()
