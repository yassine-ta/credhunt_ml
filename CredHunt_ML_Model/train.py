import argparse
import pandas as pd
from data_loader import DataLoader
from preprocessor import Preprocessor
from lcbow import LCBOW
#from fasttext_model import FastTextModel

def main(args):
    # Load data
    data_loader = DataLoader('../dataraw/snippets.csv')
    data = data_loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor()
    data = preprocessor.preprocess(data)

    # Prepare data for training
    X = data['Value']
    y = data['Positive'].astype(int).tolist()  # Use 'Positive' column as labels

    if args.model == 'lcbow':
        # Train LCBOW model
        lcbow_model = LCBOW(vocab_size=10000) # Instantiate LCBOW with vocab_size
        lcbow_model.train(X, y, epochs=args.epochs)
        #lcbow_model = LCBOW(vocab_size=10000)  # Provide a vocab_size (adjust as needed)
        #lcbow_model.train(X, y, epochs=args.epochs)
        # Save the model
        import joblib
        joblib.dump(lcbow_model, 'models/lcbow_model.pkl')
        print("LCBOW model trained and saved.")
    # elif args.model == 'fasttext':
        # Train FastText model
        # fasttext_model = FastTextModel()
        # fasttext_model.train(X, y, epochs=args.epochs)
        # Save the model
        # import joblib
        # joblib.dump(fasttext_model, 'models/fasttext_model.pkl')
        # print("FastText model trained and saved.")
    else:
        print("Invalid model choice. Please choose 'lcbow' or 'fasttext'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LCBOW or FastText model.")
    parser.add_argument('--model', type=str, required=True, choices=['lcbow', 'fasttext'], help="Model to train: 'lcbow' or 'fasttext'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    args = parser.parse_args()
    main(args)