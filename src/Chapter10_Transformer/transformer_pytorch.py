import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        """    
        Args:
            embed_dim (int): Dimensionality of the input embeddings (and output dimensions of the linear transforms).
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim 
        # Linear projection for queries, keys, and values.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, X):
        # Compute linear projections (queries, keys, values)
        Q = self.W_q(X)  # Shape: (batch_size, sequence_length, embed_dim)
        K = self.W_k(X)  # Shape: (batch_size, sequence_length, embed_dim)
        V = self.W_v(X)  # Shape: (batch_size, sequence_length, embed_dim)
        
        # Compute attention scores 
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, sequence_length, sequence_length)
        scores = scores / math.sqrt(self.embed_dim)

        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, sequence_length, sequence_length)
        # Multiply the attention weights with the values to get the final output.
        output = torch.matmul(attention_weights, V)  # Shape: (batch_size, sequence_length, embed_dim)
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        """
        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            ff_hidden_dim (int): Hidden layer dimensionality in the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention layer. We use batch_first=True so that input shape is (batch_size, sequence_length, embed_dim).
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads,
                                         batch_first=True)
        
        # First layer normalization applied after the multi-head attention residual addition.
        self.attention_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network: two linear layers with ReLU activation.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        
        # Second layer normalization after the feed-forward residual addition.
        self.ffn_norm = nn.LayerNorm(embed_dim)


    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Apply Multi-Head Attention (self-attention) where Q = K = V = x.
        # attn_output shape: (batch_size, sequence_length, embed_dim)
        attn_output = self.mha(x, x, x, need_weights=False) # need_weights=False to avoid computing the attention weights
        
        # First residual connection and layer normalization.
        # X' = LayerNorm(x + attn_output)
        x = self.attention_norm(x + attn_output)
        # Feed-Forward Network (FFN)
        ffn_output = self.ffn(x)
        # Second residual connection and layer normalization.
        # Output = LayerNorm(x + ffn_output)
        output = self.ffn_norm(x + ffn_output)
        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask, key_padding_mask)
        return x
    
# Transformer encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = encoder(src)

# Transformer decoder
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

tgt = torch.rand(10, 32, 512)
memory = torch.rand(10, 32, 512) # the output of the last layer of the encoder
out = decoder(tgt, memory)

# Transformer encoder-decoder
encoder_decoder = nn.Transformer(encoder, decoder)
src = torch.rand(10, 32, 512)
tgt = torch.rand(10, 32, 512)
out = encoder_decoder(src, tgt)

