Disease Prediction Model Comparison
Overview
This project compares the performance of 10 different machine learning classification models on a disease prediction task. The models are trained using features such as symptoms, age, gender, and other medical factors, with the goal of predicting whether a disease is present or not.

The following metrics are used for comparison:

Accuracy

Precision

Recall

F1 Score

Models are evaluated using a test dataset, and results are both displayed in tables and visualized as bar plots for easy comparison.

Models Evaluated
SVM (Support Vector Machine)

KNN (K-Nearest Neighbors)

Decision Tree

Random Forest

Naive Bayes

Logistic Regression

ANN (Artificial Neural Network)

Gradient Boosting

XGBoost

LightGBM

Key Features
Model Training: All models are trained using the preprocessed dataset. The data is split into training and testing sets to ensure reliable evaluation.

Metrics Calculation: The models are evaluated based on the following metrics:

Accuracy: Proportion of correct predictions.

Precision: Proportion of true positive predictions among all predicted positives.

Recall: Proportion of true positive predictions among all actual positives.

F1 Score: Harmonic mean of precision and recall.

Visualization: The performance of each model is visualized through bar plots to compare the metrics of different models.

CSV Output: The results are stored in a CSV file for easy access and future reference.

How to Run the Code
Install the necessary libraries: The required Python libraries include pandas, sklearn, matplotlib, and others. You can install them using pip:

bash
Copy
pip install pandas scikit-learn matplotlib xgboost lightgbm
Load the dataset: Make sure the dataset (Preprocessed_Disease_Dataset.csv) is available in your directory or adjust the file path in the code accordingly.

Run the code: Execute the Python script. The models will be trained and evaluated, and the results will be printed in the console and displayed as graphs. Additionally, the results will be saved as a CSV file named model_evaluation_comparison.csv.

bash
Copy
python disease_prediction_comparison.py
Visualize Results: After running the code, you will see the comparison metrics displayed as a series of bar plots that compare the accuracy, precision, recall, and F1 score for each model.

Files in this Repository
disease_prediction_comparison.py: The Python script that trains, evaluates, and visualizes the models.

Preprocessed_Disease_Dataset.csv: The dataset used for training and evaluation.

model_evaluation_comparison.csv: A CSV file containing the evaluation metrics for each model.

model_comparison_plots.png: The saved plot showing the comparison of model performance.

Conclusion
This project helps in understanding how different classification models perform on a disease prediction task, comparing them based on key metrics like accuracy, precision, recall, and F1 score. The visualization of these metrics aids in identifying the most suitable model for deployment based on the task requirements.







