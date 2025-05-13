# Depression Clustering Analysis Dashboard

This project is a Streamlit web application for analyzing and visualizing depression clusters among students using the Beck Depression Inventory (BDI). It provides interactive filtering, clustering insights, model comparison, and student classification features.

---

## Features
- **Interactive Dashboard**: Filter by school and program, view cluster distributions, and explore BDI score statistics.
- **Model Comparison**: Automatically trains and compares several machine learning models for clustering.
- **Student Classification**: Classify a new student into a depression cluster based on BDI responses.

---

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd janel-masters
```

### 2. Create and Activate a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

#### Using `venv` (Python 3.8+):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the Data
- Ensure the file `datasource/normalized_dataset.csv` exists. This is the main dataset used by the app.
- (Optional) `datasource/legends.txt` provides descriptions for school and program codes used in the dataset.

### 5. Run the Streamlit App
```bash
streamlit run main.py
```

### 6. Access the App in Your Browser
After running the above command, Streamlit will provide a local URL (usually [http://localhost:8501](http://localhost:8501)). Open this URL in your web browser to interact with the dashboard.

---

## File Structure
```
janel-masters/
├── datasource/
│   ├── normalized_dataset.csv   # Main dataset (required)
│   └── legends.txt              # Codebook for schools/programs (optional)
├── main.py                      # Streamlit app entry point
├── data_utils.py                # Data loading and preprocessing
├── model_utils.py               # Model training, saving, and prediction
├── requirements.txt             # Python dependencies
└── readme.md                    # This file
```

---

## Notes
- The app will automatically generate and save the best model as `student_cluster_model.pkl` after running model comparison.
- If you want to use your own data, replace `datasource/normalized_dataset.csv` with your file (ensure the format matches).
- For any issues with missing dependencies, re-run `pip install -r requirements.txt` inside your virtual environment.

---

## Troubleshooting
- **Port Already in Use**: If you get an error about port 8501, either close the other Streamlit app or run with a different port:
  ```bash
  streamlit run main.py --server.port 8502
  ```
- **Data Not Found**: Make sure `datasource/normalized_dataset.csv` exists and is not open in another program.
- **Model Not Found**: Run the model comparison section in the app to generate `student_cluster_model.pkl`.

---

## License
This project is for educational and research purposes only.
