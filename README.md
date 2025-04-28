# Smart Customer Session Predictor

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset and Feature Engineering](#dataset-and-feature-engineering)
4. [Machine Learning Models](#machine-learning-models)
5. [Key Observations and Challenges](#key-observations-and-challenges)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Running the Application Locally](#running-the-application-locally)
9. [Sample Inputs and Outputs](#sample-inputs-and-outputs)
10. [Future Work](#future-work)

## Project Overview

Understanding customer behavior is crucial for improving e-commerce business outcomes. Smart Customer Session Predictor is a Streamlit web application that predicts:

* Whether a customer will make a purchase
* The expected revenue from the session
* The behavior cluster the session falls into

It empowers businesses to prioritize high-potential customers and personalize user journeys for maximum engagement and revenue.

## Objective

The main goals of this project were:

* Build reliable classification, regression, and clustering models on session-level data.
* Automate feature engineering and data preprocessing inside the app.
* Create an easy-to-use web app where users can upload session data or fill forms manually to get instant predictions.

## Dataset and Feature Engineering

The session data consisted of raw browsing events (clicks, views, cart adds, etc.) over a period. Feature Engineering Highlights:

* Session Aggregates: Total clicks, total pageviews, total time spent.
* Price Metrics: Average product price, maximum price.
* Weekday Behavior: Flags for browsing on weekdays.
* Category and Color Preference: Encoded based on dominant categories and colors browsed.
* Country Encoding: Grouped countries into:
	+ EU Countries
	+ Non-EU Countries
	+ Germany (as special group)

All preprocessing was done automatically during both file upload and form entry.

## Machine Learning Models

We developed and optimized three separate models:

| Task | Algorithm(s) | Notes |
|------|--------------|-------|
| Purchase Prediction | Random Forest, XGBoost, etc. | Random Forest performed best after hyperparameter tuning |
| Revenue Estimation | XGBoost Regressor | Tuned for low RMSE and MAE |
| Session Clustering | KMeans Clustering | Hierarchical Clustering was considered but not used (explained below) |

⚡ Key Observations and Challenges

**Hierarchical Clustering Issues:**  
Initially, we tried Hierarchical Clustering (Agglomerative) but found it computationally expensive and unsuitable for large datasets.  
It also struggled with non-globular clusters, making KMeans a better practical choice after analyzing dendrograms and silhouette scores.

**Handling Highly Imbalanced Data:**  
The number of sessions resulting in a purchase was very low compared to non-purchase sessions.  
We tackled this using class weighting and threshold tuning rather than oversampling to avoid overfitting.

**Revenue Prediction Complexity:**  
Predicting exact revenue is inherently noisy, as many sessions have zero revenue.  
The model's focus shifted towards predicting positive revenue sessions more accurately rather than absolute precision.

**Dynamic Form Preprocessing:**  
For manual entry prediction, dynamic feature transformation was needed so that the model could accept single-session data in the same format as batch data.

🛠️ Tech Stack

- Frontend: Streamlit
- Backend: Python (Pandas, Numpy, Scikit-learn)
- ML Models: Random Forest, XGBoost, KMeans
- Deployment Ready: Local or Streamlit Cloud
- Model Storage: Joblib

📂 Project Structure

```
├── models/
│   ├── best_model_classification.pkl
│   ├── best_model_regression.pkl
│   └── best_model_clustering.pkl
├── streamlit_app.py
├── utils.py (optional if separated preprocessing)
├── README.md
├── requirements.txt
```

💻 Running the Application Locally

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/smart-customer-session-predictor.git
cd smart-customer-session-predictor
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the App**
```bash
streamlit run streamlit_app.py
```

4. **View in Browser**  
Navigate to `http://localhost:8501`

📋 Sample Inputs and Outputs

### 📄 CSV Upload Sample

| Session ID | Avg Price | Clicks | Color Group | Country | Day of Week | ... |
|------------|-----------|--------|-------------|---------|-------------|-----|
| 12345 | 49.99 | 7 | Red | Germany | Monday | ... |

### 🖥️ App Output

| Session ID | Predicted Purchase | Predicted Revenue | Cluster ID |
|------------|--------------------|-------------------|------------|
| 12345 | Yes | $56.40 | 2 |
| 67890 | No | $0.00 | 1 |

🌟 Future Work

- Add explainability using SHAP values to understand predictions better.
- Implement real-time user journey tracking.
- Optimize the app for large CSV file uploads.
- Deploy on a cloud platform (Streamlit Cloud, AWS).

