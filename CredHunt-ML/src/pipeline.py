import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CredentialDataset
from src.model import CredentialGenerator
import logging
import random
import os

class CredentialGenerationPipeline:
    def __init__(self, credentials, embed_dim=256, hidden_dim=512, num_layers=2, type_embed_dim=64, batch_size=256, learning_rate=5e-1):
        self.credentials = credentials
        self.chars = sorted(list(set(''.join([c['value'] for c in credentials]))))
        self.chars = ['<PAD>', '<START>', '<END>'] + self.chars
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        self.types = sorted(list(set([c['type'] for c in credentials])))
        self.type_to_index = {t: i for i, t in enumerate(self.types)}
        self.index_to_type = {i: t for i, t in enumerate(self.types)}
        self.num_types = len(self.types)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.type_embed_dim = type_embed_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CredentialGenerator(self.vocab_size, embed_dim, hidden_dim, num_layers, self.num_types, type_embed_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.char_to_index['<PAD>'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.num_workers = os.cpu_count()

    def prepare_data(self):
        dataset = CredentialDataset(self.credentials, self.char_to_index, self.type_to_index)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def train(self, num_epochs=10):
        self.prepare_data()
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                
                values = batch['value'].to(self.device)
                types = batch['type'].to(self.device)
                
                # Forward pass
                output = self.model(values, types)
                
                # Compute loss
                loss = self.criterion(output.view(-1, self.vocab_size), values.view(-1))
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(self.dataloader)}")
            self.scheduler.step()

    def generate_credentials(self, num_samples=5, temperature=0.8, min_length=8):
        self.model.eval()
        generated_credentials = []
        
        for _ in range(num_samples):
            # Randomly choose a credential type
            chosen_type = random.choice(self.types)
            type_index = torch.tensor([self.type_to_index[chosen_type]], dtype=torch.long).to(self.device)
            
            # Start with the start token
            input_sequence = [self.char_to_index['<START>']]
            
            # Generate the credential
            with torch.no_grad():
                while len(input_sequence) < 256:  # Limit max length
                    input_tensor = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
                    
                    # Pass the input and the chosen type to the model
                    output = self.model(input_tensor, type_index)
                    
                    # Apply temperature and get the probabilities
                    log_probs = torch.log_softmax(output[0, -1] / temperature, dim=0)
                    probabilities = torch.exp(log_probs)
                    
                    # Sample the next character
                    next_char_index = torch.multinomial(probabilities, 1).item()
                    
                    # Add the next character to the sequence
                    input_sequence.append(next_char_index)
                    
                    # If end token is generated, stop
                    if next_char_index == self.char_to_index['<END>']:
                        break
            
            # Convert the generated sequence to a string
            generated_credential = ''.join([self.index_to_char[idx] for idx in input_sequence if idx not in [0, 1, 2]])
            
            # Ensure the generated credential meets the minimum length requirement
            if len(generated_credential) >= min_length:
                generated_credentials.append(f"{generated_credential} : {chosen_type}")
        
        return generated_credentials