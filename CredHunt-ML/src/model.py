import torch
import torch.nn as nn

class CredentialGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_types, type_embed_dim, dropout=0.5):
        super(CredentialGenerator, self).__init__()
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.type_embedding = nn.Embedding(num_types, type_embed_dim)
        self.lstm = nn.LSTM(embed_dim + type_embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, value, credential_type):
        char_embedded = self.char_embedding(value)
        type_embedded = self.type_embedding(credential_type)
        type_embedded = type_embedded.unsqueeze(1).repeat(1, value.size(1), 1)
        combined_embedded = torch.cat((char_embedded, type_embedded), dim=2)
        lstm_out, _ = self.lstm(combined_embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.linear(lstm_out)
        return output