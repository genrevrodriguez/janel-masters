import pandas as pd

def useLoadData():
    df = pd.read_csv("./datasource/normalized_dataset.csv")
    df = df.dropna(subset=[col for col in df.columns if col.startswith("Q")])
    df['bdi_results'] = df['bdi_results'].astype(int)
    question_cols = [col for col in df.columns if col.startswith("Q") and col[1:].isdigit()]
    if "Cluster" in df.columns:
        df_features = df[question_cols]
        df_target = df["Cluster"]

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    question_cols = [col for col in df.columns if col.strip().startswith("Q") and col.strip()[1:].isdigit()]
    df[question_cols] = df[question_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=question_cols)

    if "Cluster" not in df.columns:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[question_cols])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

    return df 