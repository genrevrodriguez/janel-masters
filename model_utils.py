import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

def useTrainAndSaveBestModel(X, y, question_cols, model_path="student_cluster_model.pkl"):
    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "LightGBM": LGBMClassifier(verbose=-1, min_child_samples=1, min_split_gain=0.0),
        "ANN (MLP)": MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
        "Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(max_iter=300)
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    best_acc = 0
    best_model = None
    best_model_name = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results.append({"Model": name, "Accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
    joblib.dump(best_model, model_path)
    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False), best_model_name

def useLoadBestModel(model_path="student_cluster_model.pkl"):
    return joblib.load(model_path)

def useClassifyStudent(bdi_responses, model, question_cols):
    X_new = np.array([bdi_responses[col] for col in question_cols]).reshape(1, -1)
    return model.predict(X_new)[0] 