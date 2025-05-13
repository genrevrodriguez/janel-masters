import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import useLoadData
from model_utils import useTrainAndSaveBestModel, useLoadBestModel, useClassifyStudent

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
    cluster_counts = filtered_df["Cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Show average BDI per cluster
    avg_bdi = filtered_df.groupby("Cluster")["bdi_results"].mean()
    st.write("**Average BDI per Cluster**")
    st.dataframe(avg_bdi.reset_index().rename(columns={"bdi_results": "Average BDI Score"}))

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
    with st.form("student_form"):
        bdi_responses = {}
        for col in question_cols:
            bdi_responses[col] = st.number_input(f"{col}", min_value=0, max_value=3, value=0)
        submitted = st.form_submit_button("Classify")
        if submitted:
            cluster = useClassifyStudent(bdi_responses, model, question_cols)
            st.success(f"The student is classified into Cluster {cluster}")
except Exception as e:
    st.info("Train and save a model first to enable student classification.")