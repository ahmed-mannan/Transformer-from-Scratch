#input embedding - Original text gets converted into a vector representation using an embedding layer. The embedding layer maps each token in the input sequence to a dense vector of fixed size (d_model). The output of the embedding 
# layer is then scaled by the square root of d_model to maintain the variance of the embeddings. Vector size of 512.

import math

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size: int):
        super().__init__()
        self.d_model = d_model #512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


#Positional encoding - Since the transformer architecture does not have any inherent notion of the order of the input tokens, positional encodings are added to the input embeddings to provide information about the position of each token in the sequence. The positional encodings are generated using sine and cosine functions of different frequencies, which allows the model to learn the relative positions of tokens in the sequence. The positional encoding matrix has a shape of (seq_len, d_model), where seq_len is the maximum 
# length of the input sequence and d_model is the size of the embedding vectors. The positional encodings are added to the input embeddings before being fed into the transformer layers.
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len,) containing the position indices
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #(seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #(d_model/2,)
       
        #Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        #Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
       
        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        x = x + (self.pe[:, :x.shape[1], :]).require_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps #we need e for numerical stability
        self.alpha = nn.Parameter(torch.ones(1)) #scaling factor , multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #bias term, added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) #W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) #W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
