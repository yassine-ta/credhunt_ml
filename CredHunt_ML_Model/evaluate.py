# src/evaluate.py
import pandas as pd
from data_loader import DataLoader
from preprocessor import Preprocessor
from lcbow import LCBOW
from fasttext_model import FastTextModel
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data_loader = DataLoader('data/raw/snippets.csv')
data = data_loader.load_data()

# Preprocess data
preprocessor = Preprocessor()
data = preprocessor.preprocess(data)

# Prepare data for evaluation
X = data['cleaned_snippet']
y = data['Type'].apply(lambda x: 1 if x == 'Password' else 0)  # Example: binary classification for Password type

# Load the trained models
lcbow_model = joblib.load('models/lcbow_model.pkl')
fasttext_model = joblib.load('models/fasttext_model.pkl')

# Evaluate LCBOW model
lcbow_predictions = lcbow_model.predict(X)

# Evaluate FastText model
fasttext_predictions = fasttext_model.predict(X)

# Calculate evaluation metrics
def evaluate_model(predictions, y):
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    return accuracy, precision, recall, f1

# Evaluate LCBOW model
lcbow_accuracy, lcbow_precision, lcbow_recall, lcbow_f1 = evaluate_model(lcbow_predictions, y)
print("LCBOW Model Evaluation:")
print(f'Accuracy: {lcbow_accuracy}')
print(f'Precision: {lcbow_precision}')
print(f'Recall: {lcbow_recall}')
print(f'F1 Score: {lcbow_f1}')

# Evaluate FastText model
fasttext_accuracy, fasttext_precision, fasttext_recall, fasttext_f1 = evaluate_model(fasttext_predictions, y)
print("\nFastText Model Evaluation:")
print(f'Accuracy: {fasttext_accuracy}')
print(f'Precision: {fasttext_precision}')
print(f'Recall: {fasttext_recall}')
print(f'F1 Score: {fasttext_f1}')