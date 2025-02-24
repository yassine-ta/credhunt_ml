# src/preprocessor.py
import re

class Preprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text

    def preprocess(self, data):
        data['cleaned_snippet'] = data['Snippet Code'].apply(self.clean_text)
        return data