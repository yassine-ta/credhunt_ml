# src/fasttext_model.py
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

class FastTextModel:
    def __init__(self):
        self.model = None

    def train(self, X, y, epochs=10):
        # Prepare data for FastText
        data = pd.DataFrame({'label': y, 'text': X})
        data['label'] = '__label__' + data['label'].astype(str)
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data.to_csv('data/processed/train.txt', index=False, sep=' ', header=False)
        val_data.to_csv('data/processed/val.txt', index=False, sep=' ', header=False)

        # Train FastText model
        self.model = fasttext.train_supervised('data/processed/train.txt', epoch=epochs)

    def predict(self, X):
        predictions = [self.model.predict(text)[0][0].replace('__label__', '') for text in X]
        return predictions