import torch
import torch.nn as nn
from einops import einsum
from typing import Optional


class Linear(nn.Module):
    """
    Linear transformation module that performs y = xW^T.
    
    This implementation follows PyTorch's nn.Linear interface but without bias.
    """
    
    def __init__(self, in_features: int, 
                 out_features: int, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the Linear module.
        
        Args:
            in_features: Size of each input sample (final dimension)
            out_features: Size of each output sample (final dimension)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2.0 / (in_features + out_features)) ** 0.5
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        # return torch.matmul(x, self.weight.t())
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """
    Embedding lookup module that maps token IDs to embedding vectors.
    
    This implementation follows PyTorch's nn.Embedding interface.
    """
    
    def __init__(self, num_embeddings: int, 
                 embedding_dim: int, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the Embedding module.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix as a parameter
        # Shape: (num_embeddings, embedding_dim) with d_model as final dimension
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.embedding, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for the given token IDs.
        
        Args:
            token_ids: Input tensor of token IDs of shape [batch_size, sequence_length]
            
        Returns:
            Embedding vectors of shape [..., embedding_dim]
        """
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization module.
    
    RMSNorm normalizes the input using the root mean square of the elements,
    then applies a learnable affine transformation.
    """
    
    def __init__(self, d_model: int, 
                 eps: float = 1e-5, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        in_dtype = x.dtype
        x = x.to(torch.float32) # prevent overflow for x.pow(2)
        
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # (batch_size, sequence_length, 1)
        x_norm = x / rms
        # out = x_norm * self.weight
        out = einsum(x_norm, self.weight, 'b t d, d->b t d')
        return out.to(in_dtype)
        
        
        
        