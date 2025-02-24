import pandas as pd
import joblib
import torch
from nltk.tokenize import word_tokenize
from lcbow import LCBOW, TextDataset, collate_batch  # Import LCBOW and helper functions

def load_test_data(file_path):
    """Loads the test data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_test_data(df, word_to_index):
    """Preprocesses the test data using the same vocabulary as the training data."""
    texts = df['Value'].tolist()
    labels = df['Positive'].astype(int).tolist()  # Use 'Positive' column as labels

    # Ensure all texts are strings
    texts = [str(text) for text in texts]

    # Tokenize and convert to indices, handling unknown words
    tokenized_texts = []
    for text in texts:
        tokens = [word_to_index.get(word, 0) for word in word_tokenize(text)]  # Use .get() to handle unknown words
        if not tokens:
            tokens = [0]  # Use padding if no tokens are found
        tokenized_texts.append(tokens)

    return tokenized_texts, labels

def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test data."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device).float()

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def main():
    # Load the trained model
    model_path = 'models/lcbow_model.pkl'
    model = joblib.load(model_path)
    model.eval()

    # Load the test data
    test_file_path = '../dataraw/data_test.csv'
    test_df = load_test_data(test_file_path)

    # Create word_to_index mapping from the training data (assuming it's stored in the model)
    word_to_index = {"<PAD>": 0}  # Add padding token
    index_to_word = {0: "<PAD>"}
    vocab_size = 1
    for text in test_df['Value']:
        for word in word_tokenize(text):
            if word not in word_to_index:
                word_to_index[word] = vocab_size
                index_to_word[vocab_size] = word
                vocab_size += 1

    # Preprocess the test data
    test_texts, test_labels = preprocess_test_data(test_df, word_to_index)

    # Create a test dataset and dataloader
    test_dataset = TextDataset(test_texts, test_labels, word_to_index)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=collate_batch)

    # Define the loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Determine the device to use (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluate the model
    avg_loss, accuracy = evaluate_model(model, test_loader, criterion, device)

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()