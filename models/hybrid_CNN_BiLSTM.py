"""
        Gang Liu, Jiabao Guo,
        Bidirectional LSTM with attention mechanism and convolutional layer for text classification,
        Neurocomputing,
        Volume 337,
        2019,
        Pages 325-338,
        ISSN 0925-2312,
        https://doi.org/10.1016/j.neucom.2019.01.078.
        """
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, hidden_dim, output_dim, dropout=0.5):
        super(HybridCNNBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.embedding_dropout = nn.Dropout2d(0.2) 
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs, padding='same'),
                nn.BatchNorm1d(n_filters), 
                nn.ReLU()
            ) for fs in filter_sizes
        ])
        
        self.lstm = nn.LSTM(n_filters * len(filter_sizes), 
                            hidden_dim, 
                            bidirectional=True, 
                            batch_first=True, 
                            num_layers=1,
                            dropout=dropout if dropout > 0 else 0) 
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text) 
        
        embedded = embedded.permute(0, 2, 1) 
        embedded = self.embedding_dropout(embedded.unsqueeze(2)).squeeze(2)
        
        conved = [conv(embedded) for conv in self.convs]
        combined = torch.cat(conved, dim=1).permute(0, 2, 1)
        
        lstm_out, (hidden, _) = self.lstm(combined)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))