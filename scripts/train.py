import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore")

# --------------- Load Dataset ---------------
train_data = pd.read_csv("E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/processedData/train_data_processed.csv")

# --------------- Classification Setup ---------------
def classification_training(train_data):
    print("🔵 Starting Classification Training...")
    
    X = train_data.drop(['purchase', 'revenue', 'session_id'], axis=1)
    y = train_data['purchase']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Neural Network": MLPClassifier(max_iter=300, random_state=42)
    }
    
    param_grids = {
        "Logistic Regression": {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        },
        "Decision Tree": {
            'criterion': ['gini', 'entropy'],
            'max_depth': [5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'splitter': ['best', 'random']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        "XGBoost": {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'n_estimators': [100, 200, 300],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_lambda': [1, 5, 10]
        },
        "Neural Network": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [300]
        }
    }

    best_models = {}
    results = []
    
    for model_name, model in models.items():
        print(f"\n🔵 Tuning {model_name}...")
        
        rscv = RandomizedSearchCV(
            model,
            param_distributions=param_grids[model_name],
            n_iter=10,
            cv=2,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        rscv.fit(X_train, y_train)
        best_model = rscv.best_estimator_
        y_pred = best_model.predict(X_test)
        
        best_models[model_name] = best_model
        
        results.append({
            "Model": model_name,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        })
        
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")
    
    results_df = pd.DataFrame(results)
    print("\n🔵 Tuned Model Comparison:\n", results_df.sort_values(by="ROC-AUC", ascending=False))
    
    # Save best model
    best_model_name = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]['Model']
    joblib.dump(best_models[best_model_name], 'E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/models/best_model_classification.pkl')
    
    print(f"\n✅ Best classification model saved: {best_model_name}")

# --------------- Regression Setup (similar format) ---------------
def regression_training(train_data):
    print("🔵 Starting Regression Training...")
    
    X = train_data.drop(['purchase', 'revenue', 'session_id'], axis=1)
    y = train_data['revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Define models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    # 3. Hyperparameter grids
    param_grids = {
        'Ridge': {
            'alpha': [0.01, 0.1, 1, 10, 50, 100]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }

    results = []
    best_models = {}

    # 4. Training Loop
    for name, model in models.items():
        print(f"\n🔵 Training {name}...")

        if name in param_grids:
            grid = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[name],
                n_iter=5,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best Parameters for {name}: {grid.best_params_}")

        elif name == 'GradientBoosting':
            best_model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            )
            best_model.fit(X_train, y_train)

        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Save trained model
        best_models[name] = best_model

        # Evaluate
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })

    results_df = pd.DataFrame(results)
    print("\n🔵 Regression Model Comparison:\n", results_df.sort_values(by="R2", ascending=False))

    best_model_name = results_df.sort_values('R2', ascending=False).iloc[0]['Model']
    best_model = best_models[best_model_name]
    
    joblib.dump(best_model, 'E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/models/best_model_regression.pkl')
    print(f"\n✅ Best regression model saved:- {best_model_name}")

# --------------- Clustering Setup (basic KMeans) ---------------
def clustering_training(train_data):
    print("\n🔵 Starting Clustering Training...\n")

    # 1. Prepare Data
    X = train_data.drop(['purchase', 'revenue', 'session_id'], axis=1)
    print(f"Shape of data used for clustering: {X.shape}")

    # Optional: PCA to reduce dimensionality and speed up
    """pca = PCA(n_components=0.95, random_state=42)  # 95% variance retained
    X_reduced = pca.fit_transform(X)
    print(f"Shape after PCA: {X_reduced.shape}")"""

    # 2. KMeans with k=4
    print("\n🔵 Training KMeans Clustering (k=2)...")

    kmeans_final = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X)

    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    print(f"✅ KMeans Silhouette Score (k=2): {kmeans_silhouette:.4f}")

    # 3. DBSCAN
    print("\n🔵 Training DBSCAN Clustering...")

    dbscan = DBSCAN(eps=2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    if n_clusters_dbscan > 1:
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
        print(f"✅ DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
        print(f"✅ DBSCAN found {n_clusters_dbscan} clusters (excluding noise).")
    else:
        dbscan_silhouette = -1  # invalid
        print("⚠️ DBSCAN did not form multiple clusters. Ignoring DBSCAN.")

    # 4. Choose Best Model
    if kmeans_silhouette >= dbscan_silhouette:
        best_model = kmeans_final
        best_labels = kmeans_labels
        best_algo = "KMeans"
        best_silhouette = kmeans_silhouette
    else:
        best_model = dbscan
        best_labels = dbscan_labels
        best_algo = "DBSCAN"
        best_silhouette = dbscan_silhouette

    # Save best model
    joblib.dump(best_model, 'E:/AI engineer/Guvi/Capstone Projects/Project4/Customer_Conversion_Analysis/models/best_model_clustering.pkl')
    print(f"\n🏆 Best Model: {best_algo} with Silhouette Score: {best_silhouette:.4f}")

    # 5. Attach cluster labels to data
    train_data['cluster_label'] = best_labels

# --------------- Main Program ---------------
if __name__ == "__main__":
    print("\n🚀 Training Started...")
    
    #classification_training(train_data)
    #regression_training(train_data)
    clustering_training(train_data)
    
    print("\n🎯 All Models Trained and Saved Successfully!")
