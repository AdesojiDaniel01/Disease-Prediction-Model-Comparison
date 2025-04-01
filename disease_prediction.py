# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('Preprocessed_Disease_Dataset.csv')  # Dataset name updated

# Step 2: Show the first few rows of the dataset
print(data.head())

# Step 3: Define features (X) and target (y)
X = data.drop(columns=['Disease', 'Outcome Variable'])  # Adjust columns based on dataset
y = data['Outcome Variable']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize models
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'ANN': MLPClassifier(max_iter=500),
    'Linear Regression': LinearRegression(),  # Regression model
    'K-Means Clustering': KMeans(n_clusters=3),  # Clustering model
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)  # Clustering model
}

# Step 6: Train, predict, and evaluate each model
evaluation_metrics = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set if it's a classification model
    if name not in ['K-Means Clustering', 'DBSCAN']:  # Clustering models don't need test labels
        y_pred = model.predict(X_test)
    else:
        y_pred = model.labels_  # Get clustering labels
    
    # Calculate metrics for classification models
    if name not in ['K-Means Clustering', 'DBSCAN']:  # Skip metrics calculation for clustering
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        evaluation_metrics[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    else:
        # For clustering, we can only print the labels (clustering doesn't use y_test)
        evaluation_metrics[name] = {
            'Cluster Labels': y_pred
        }

# Step 7: Display evaluation metrics for classification models
classification_metrics = {key: value for key, value in evaluation_metrics.items() if key not in ['K-Means Clustering', 'DBSCAN']}

classification_df = pd.DataFrame(classification_metrics).T
print("Classification Model Evaluation Metrics:")
print(classification_df)

# Step 8: Visualize model performance (optional)
models_classification = [model for model in models if model not in ['K-Means Clustering', 'DBSCAN']]
accuracies = [evaluation_metrics[model]['Accuracy'] for model in models_classification]

plt.bar(models_classification, accuracies)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()

# Step 9: Save the evaluation metrics to a CSV file for future reference
classification_df.to_csv('model_evaluation.csv')
