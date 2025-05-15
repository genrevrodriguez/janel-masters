import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_data():
    return pd.read_csv("./datasource/labeled_dataset.csv")

def get_user_input(possible_values):
    input_dict = {}
    for feature, values in possible_values.items():
        if feature == "age":
            input_dict[feature] = st.number_input("Age", min_value=10, max_value=100, value=20)
        else:
            input_dict[feature] = st.selectbox(feature.replace("_", " ").title(), sorted(values), key=feature)
    input_df = pd.DataFrame([input_dict])
    return input_df

def main():
    st.title("Depression Cluster Predictor (PHQ & BDI)")
    df = load_data()

    # Feature selection
    all_features = [
        "sex", "age", "school_name", "program", "department", "year_level",
        "ses", "family_arrangement", "living_condition_level"
    ]
    default_features = [
        "sex", "age", "school_name", "ses", "family_arrangement", "living_condition_level"
    ]
    selected_features = st.multiselect(
        "Select features to include in training and prediction:",
        options=all_features,
        default=default_features
    )
    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    def train_models_selected(df, selected_features):
        X = df[selected_features]
        y_phq = df["phq_cluster"]
        y_bdi = df["bdi_cluster"]

        X_encoded = pd.get_dummies(X)
        X_train, _, y_phq_train, _, y_bdi_train, _ = train_test_split(
            X_encoded, y_phq, y_bdi, test_size=0.2, random_state=42
        )

        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "LightGBM": LGBMClassifier(verbose=-1, min_child_samples=1, min_split_gain=0.0),
            "ANN (MLP)": MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
            "Discriminant Analysis": LinearDiscriminantAnalysis(),
            "Logistic Regression": LogisticRegression(max_iter=300)
        }

        trained_models = {}
        for name, model in models.items():
            model_phq = model
            model_bdi = model.__class__(**model.get_params())
            model_phq.fit(X_train, y_phq_train)
            model_bdi.fit(X_train, y_bdi_train)
            trained_models[name] = {"phq": model_phq, "bdi": model_bdi}

        return trained_models, X_encoded.columns

    trained_models, model_features = train_models_selected(df, selected_features)

    algo_choice = st.selectbox("Choose Model", list(trained_models.keys()))
    st.markdown("### Input Information")
    user_input = get_user_input({col: df[col].unique() for col in selected_features})

    # Preprocess
    user_input_encoded = pd.get_dummies(user_input)
    full_input = pd.DataFrame(columns=model_features)
    full_input.loc[0] = 0
    for col in user_input_encoded.columns:
        if col in full_input.columns:
            full_input.at[0, col] = user_input_encoded.at[0, col]

    if st.button("Predict"):
        model_set = trained_models[algo_choice]
        pred_phq = model_set["phq"].predict(full_input)[0]
        pred_bdi = model_set["bdi"].predict(full_input)[0]

        phq_map = [
            "Minimal or No Depression",
            "Mild Depression",
            "Moderate Depression",
            "Moderately Severe Depression",
            "Severe Depression"
        ]
        bdi_map = ["Minimal Depression", "Mild Depression", "Moderate Depression", "Severe Depression"]

        st.success(f"**PHQ Cluster**: {pred_phq} - {phq_map[pred_phq]}")
        st.success(f"**BDI Cluster**: {pred_bdi} - {bdi_map[pred_bdi]}")

if __name__ == "__main__":
    main()