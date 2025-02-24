import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Download punkt tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_index):
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        tokens = [self.word_to_index[word] for word in word_tokenize(text) if word in self.word_to_index]
        # Convert tokens to tensor

        # Check if tokens is empty
        if not tokens:
            tokens = [0]  # Use a padding index if the sequence is empty

        return torch.tensor(tokens), torch.tensor(label)

def collate_batch(batch):
    # Separate texts and labels
    texts, labels = zip(*batch)

    # Pad sequences
    texts = [torch.tensor(text) for text in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # Use padding_value=0

    # Convert labels to tensor
    labels = torch.tensor(labels)

    return texts_padded, labels


class LCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super(LCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)  # Output a single value for binary classification
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        embedded = self.embeddings(x)
        pooled = torch.mean(embedded.float(), dim=1)  # Average the embeddings
        return torch.sigmoid(self.linear(pooled)).squeeze(1)  # Apply sigmoid for binary classification

    def train(self, texts, labels, epochs=10, learning_rate=0.01): # Reduced learning rate
        # Build vocabulary
        word_to_index = {"<PAD>": 0}  # Add padding token
        index_to_word = {0: "<PAD>"}
        vocab_size = 1
        for text in texts:
            for word in word_tokenize(text):
                if word not in word_to_index:
                    word_to_index[word] = vocab_size
                    index_to_word[vocab_size] = word
                    vocab_size += 1

        # Create dataset and dataloader
        dataset = TextDataset(texts, labels, word_to_index)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

        # Define model, loss function, and optimizer
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, 128)  # Initialize embedding layer here
        self.linear = nn.Linear(128, 1)
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss with logits
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        super(LCBOW, self).train() # call the nn.Module train method
        for epoch in range(epochs):
            correct = 0
            total = 0
            for texts, labels in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(texts)  # Squeeze to remove extra dimensions
                labels = labels.float()  # Convert labels to float
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    def train(self, mode=True):
        """Sets the module in training mode."""
        super().train(mode)