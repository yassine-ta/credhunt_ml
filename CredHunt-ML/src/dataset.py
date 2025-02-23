import torch
from torch.utils.data import Dataset

class CredentialDataset(Dataset):
    def __init__(self, credentials, char_to_index, type_to_index, max_length=128):
        self.char_to_index = char_to_index
        self.type_to_index = type_to_index
        self.max_length = max_length
        
        # Preload data into memory
        self.value_indices = []
        self.type_indices = []
        for credential in credentials:
            value = credential['value']
            credential_type = credential['type']

            # Convert value to indices
            value_indices = [self.char_to_index[char] for char in value]
            value_indices = value_indices[:self.max_length]  # Truncate if necessary
            value_indices += [self.char_to_index['<PAD>']] * (self.max_length - len(value_indices))  # Pad
            self.value_indices.append(torch.tensor(value_indices, dtype=torch.long))

            # Convert type to index
            type_index = self.type_to_index[credential_type]
            self.type_indices.append(torch.tensor(type_index, dtype=torch.long))

    def __len__(self):
        return len(self.value_indices)

    def __getitem__(self, idx):
        return {
            'value': self.value_indices[idx],
            'type': self.type_indices[idx]
        }