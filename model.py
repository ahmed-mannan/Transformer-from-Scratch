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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model) #wq
        self.w_k = nn.Linear(d_model, d_model) #wk
        self.w_v = nn.Linear(d_model, d_model) #wv

        self.w_o = nn.Linear(d_model, d_model) #wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float('-inf'))  # Apply the mask to the attention scores
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores  # (batch, num_heads, seq_len, d_k), (batch, num_heads, seq_len, seq_len)
    
    def forward(self, query, key, value, mask):
        query = self.w_q(query)  # (batch, seq_len, d_model)
        key = self.w_k(key)      # (batch, seq_len, d_model)
        value = self.w_v(value)  # (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)  
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) 

        return self.w_o(x)  # (batch, seq_len, d_model)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Apply layer normalization before the sublayer and add the residual connection


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  # Two residual connections, one for self-attention and one for feed-forward

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # Self-attention with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)  # Feed-forward with residual connection
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # Final layer normalization after all encoder blocks

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # Pass the input through each encoder block
        return self.norm(x)  # Apply final layer normalization