import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import useLoadData
from model_utils import useTrainAndSaveBestModel, useLoadBestModel, useClassifyStudent
import json

def useLoadQuestions():
    with open("questions.json", "r") as f:
        return json.load(f)

# Load the cleaned dataset
@st.cache_data
def load_data():
    return useLoadData()

df = load_data()

# Sidebar filter
st.sidebar.title("Filter Options")
selected_school = st.sidebar.multiselect("Select School", options=sorted(df["school_name"].unique()), default=None)
selected_program = st.sidebar.multiselect("Select Program", options=sorted(df["program"].unique()), default=None)

filtered_df = df.copy()
if selected_school:
    filtered_df = filtered_df[filtered_df["school_name"].isin(selected_school)]
if selected_program:
    filtered_df = filtered_df[filtered_df["program"].isin(selected_program)]

# Main content
st.title("Depression Clustering Analysis")
st.write("This dashboard shows insights from BDI (Beck Depression Inventory) clustering.")

# Cluster Summary
if "Cluster" in filtered_df.columns:
    st.subheader("BDI Cluster Distribution")
    # Define cluster labels
    cluster_labels = {
        0: "Severe depression",
        1: "Mild/Moderate depression",
        2: "Minimal/No depression"
    }

    # Add labels to cluster counts
    cluster_counts = filtered_df["Cluster"].value_counts().sort_index()
    cluster_counts.index = [f"{i} ({cluster_labels.get(i, 'Unknown')})" for i in cluster_counts.index]
    st.bar_chart(cluster_counts)

    # Show average BDI per cluster with labels
    avg_bdi = filtered_df.groupby("Cluster")["bdi_results"].mean()
    avg_bdi_labeled = avg_bdi.reset_index()
    avg_bdi_labeled["Label"] = avg_bdi_labeled["Cluster"].map(cluster_labels)
    avg_bdi_labeled["Cluster"] = avg_bdi_labeled.apply(
        lambda row: f"{row['Cluster']} ({row['Label']})", axis=1
    )
    avg_bdi_labeled = avg_bdi_labeled.drop(columns=["Label"])
    st.write("**Average BDI per Cluster**")
    st.dataframe(avg_bdi_labeled.rename(columns={"bdi_results": "Average BDI Score"}))

    # BDI Score Histogram
    st.subheader("BDI Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["bdi_results"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("BDI Score")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)
else:
    st.warning("No cluster information found in the dataset.")

st.subheader("Model Comparison")

# Define features and target
question_cols = [col for col in filtered_df.columns if col.startswith("Q") and col[1:].isdigit()]
if "Cluster" in filtered_df.columns and all(col in filtered_df.columns for col in question_cols):
    X = filtered_df[question_cols]
    y = filtered_df["Cluster"]
    results_df, best_model_name = useTrainAndSaveBestModel(X, y, question_cols)
    st.dataframe(results_df)
    st.info(f"Best model '{best_model_name}' saved for student classification.")
else:
    st.warning("Cannot run model comparison: missing Cluster column or BDI question data.")

# --- Student Classification Section ---
st.subheader("Classify a New Student")
try:
    model = useLoadBestModel()
    questions = useLoadQuestions()
    with st.form("student_form"):
        bdi_responses = {}
        for col in question_cols:
            options = list(questions[col].keys())
            option_labels = [f"{val}: {questions[col][val]}" for val in options]
            selected = st.selectbox(f"{col}", options=options, format_func=lambda x: questions[col][x])
            bdi_responses[col] = int(selected)
        submitted = st.form_submit_button("Classify")
        if submitted:
            cluster = useClassifyStudent(bdi_responses, model, question_cols)
            # Add cluster label mapping
            cluster_labels = {
                0: "Severe depression",
                1: "Mild/Moderate depression",
                2: "Minimal/No depression"
            }
            label = cluster_labels.get(cluster, "Unknown")
            st.success(f"The student is classified into Cluster {cluster} ({label})")
except Exception as e:
    st.info("Train and save a model first to enable student classification.")