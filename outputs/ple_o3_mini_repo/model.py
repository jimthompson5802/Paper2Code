"""
model.py

This module defines the Model class and its submodules that implement
numerical feature embeddings and backbone architectures for tabular deep learning.
It supports three embedding modules:
  - BaselineEmbedding: A simple linear mapping.
  - PiecewiseLinearEmbedding (PLE): Computes a piecewise linear encoding using precomputed bin boundaries.
  - PeriodicEmbedding: Applies a periodic activation (sin(k * x + c)) followed by a linear layer.

It also implements three backbone networks:
  - MLPBackbone: A multi-layer perceptron.
  - ResNetBackbone: A ResNet-style model with residual blocks.
  - TransformerBackbone: A Transformer encoder that treats per-feature embeddings as tokens.

The main Model class wires these submodules together following the "Backbone-Embedding" pattern.
All hyperparameters are read from the provided configuration dictionary (from config.yaml).

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Embedding Submodules
# ---------------------------

class BaselineEmbedding(nn.Module):
    """
    Baseline embedding module: a simple linear mapping with an activation.
    
    Input:
      x: tensor of shape [batch_size] (a single feature scalar per sample).
    Output:
      Tensor of shape [batch_size, output_dim].
    """
    def __init__(self, output_dim: int = 64) -> None:
        super(BaselineEmbedding, self).__init__()
        self.linear = nn.Linear(1, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x has shape [B, 1]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = self.linear(x)
        out = self.activation(out)
        return out

class PiecewiseLinearEmbedding(nn.Module):
    """
    Piecewise Linear Encoding (PLE) embedding module.
    
    This module implements a piecewise linear interpolation of a scalar input x
    based on precomputed bin boundaries. Given bin boundaries b0, b1, ..., bT,
    it computes an activation vector e of length T as follows for each bin j:
      - If x < b_j (for j==0, if x < b0): e[j] = 0.
      - If x >= b_{j+1}: e[j] = 1.
      - If x is inside [b_j, b_{j+1}]:
            e[j] = (x - b_j) / (b_{j+1} - b_j).
    Finally, the output embedding is computed as:
         f(x) = v0 + (e dot v)
    where v0 (bias) and v (transformation weights) are learnable parameters.
    
    Input:
      x: tensor of shape [batch_size] (a single feature scalar per sample).
    Output:
      Tensor of shape [batch_size, output_dim].
    """
    def __init__(self, output_dim: int = 64, bin_boundaries: torch.Tensor = None) -> None:
        super(PiecewiseLinearEmbedding, self).__init__()
        # Use provided bin boundaries; if None, default to two boundaries [0, 1]
        if bin_boundaries is None:
            bin_boundaries = torch.tensor([0.0, 1.0], dtype=torch.float32)
        # Store bin boundaries as a buffer so they are not updated by gradient descent
        self.register_buffer("bin_boundaries", bin_boundaries.float())
        # Number of bins = (# boundaries) - 1
        self.num_bins: int = self.bin_boundaries.shape[0] - 1
        self.output_dim: int = output_dim
        # Learnable parameters: bias v0 and weight matrix v of shape [num_bins, output_dim]
        self.v0 = nn.Parameter(torch.zeros(output_dim))
        self.v = nn.Parameter(torch.randn(self.num_bins, output_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size]
        x_exp = x.unsqueeze(1)  # shape: [B, 1]
        # Prepare left and right bin boundaries: shape [1, num_bins]
        b_left = self.bin_boundaries[:-1].unsqueeze(0)
        b_right = self.bin_boundaries[1:].unsqueeze(0)
        # Compute piecewise activation vector e for each sample and each bin:
        # If x < b_left, e=0; if x >= b_right, e=1; else, linear interpolation.
        e = torch.where(
                x_exp < b_left,
                torch.zeros_like(x_exp),
                torch.where(
                    x_exp >= b_right,
                    torch.ones_like(x_exp),
                    (x_exp - b_left) / (b_right - b_left + 1e-6)
                )
            )  # shape: [B, num_bins]
        # Compute weighted sum: e dot v, then add bias v0.
        out = self.v0 + torch.matmul(e, self.v)  # shape: [B, output_dim]
        return out

class PeriodicEmbedding(nn.Module):
    """
    Periodic embedding module.
    
    Applies a periodic transformation to a scalar input x using the formula:
         z = k * x + c
         y = sin(z)
         f(x) = Linear(y)
    where:
         - k is a tunable multiplier.
         - c is a trainable bias parameter initialized from N(0, sigma).
         - Linear() is a trainable linear layer mapping the scalar periodic output to the embedding space.
    
    Input:
      x: tensor of shape [batch_size] (a single feature scalar per sample).
    Output:
      Tensor of shape [batch_size, output_dim].
    """
    def __init__(self, output_dim: int = 64, k: int = 16, sigma: float = 0.1) -> None:
        super(PeriodicEmbedding, self).__init__()
        self.k: float = float(k)
        # Trainable parameter c initialized using Normal(0, sigma)
        self.c = nn.Parameter(torch.randn(1) * sigma)
        self.linear = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size]
        x_exp = x.unsqueeze(1)  # shape: [B, 1]
        z = self.k * x_exp + self.c  # shape: [B, 1]
        y = torch.sin(z)            # shape: [B, 1]
        out = self.linear(y)        # shape: [B, output_dim]
        return out

# ---------------------------
# Backbone Modules
# ---------------------------

class MLPBackbone(nn.Module):
    """
    MLP backbone for aggregating flattened embeddings.
    
    Architecture:
      - A feed-forward network with configurable number of layers and hidden dimension.
      - Activation: ReLU between layers.
      - Final layer produces the model prediction (e.g., regression scalar or class logits).
    
    Input:
      Tensor of shape [batch_size, num_features * embedding_output_dim].
    Output:
      Tensor of shape [batch_size, output_dim] (default output_dim: 1).
    """
    def __init__(self, input_dim: int, output_dim: int = 1, config: dict = None) -> None:
        super(MLPBackbone, self).__init__()
        # Use configuration parameters with default fallbacks.
        hidden_dim = config.get("mlp", {}).get("hidden_dim", 256) if config is not None else 256
        num_layers = config.get("mlp", {}).get("num_layers", 4) if config is not None else 4
        layers = []
        in_dim = input_dim
        # Build hidden layers with ReLU activation
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ResidualBlock(nn.Module):
    """
    A basic residual block used in the ResNet backbone.
    """
    def __init__(self, dim: int) -> None:
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + identity
        out = self.activation(out)
        return out

class ResNetBackbone(nn.Module):
    """
    ResNet backbone for tabular data.
    
    Architecture:
      - Input layer projects flattened embeddings to hidden dimension.
      - Several residual blocks with skip connections.
      - Final linear layer for prediction.
    
    Input:
      Tensor of shape [batch_size, num_features * embedding_output_dim].
    Output:
      Tensor of shape [batch_size, output_dim].
    """
    def __init__(self, input_dim: int, output_dim: int = 1, config: dict = None) -> None:
        super(ResNetBackbone, self).__init__()
        hidden_dim = config.get("mlp", {}).get("hidden_dim", 256) if config is not None else 256
        num_layers = config.get("resnet", {}).get("num_layers", 4) if config is not None else 4
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        # Create (num_layers - 2) residual blocks.
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.input_layer(x))
        for block in self.residual_blocks:
            out = block(out)
        out = self.output_layer(out)
        return out

class TransformerBackbone(nn.Module):
    """
    Transformer backbone for tabular data.
    
    Architecture:
      - Treats each numerical feature's embedding as a token.
      - Applies a token embedding layer to inject feature index information.
      - Uses a Transformer encoder stack.
      - Aggregates token information via mean pooling.
      - Final linear layer outputs the prediction.
    
    Input:
      Tensor of shape [batch_size, num_features, embedding_dim].
    Output:
      Tensor of shape [batch_size, output_dim].
    """
    def __init__(self, num_tokens: int, embedding_dim: int, output_dim: int = 1, config: dict = None) -> None:
        super(TransformerBackbone, self).__init__()
        num_layers = config.get("transformer", {}).get("num_layers", 2) if config is not None else 2
        nhead = 4  # Default number of attention heads
        dropout = 0.1  # Default dropout rate
        # Linear layer to inject feature index information
        self.token_embedding = nn.Linear(embedding_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pooling: average pooling across tokens
        self.output_layer = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, num_tokens, embedding_dim]
        # Inject token-level information
        x = self.token_embedding(x)
        # Transformer encoder expects input shape [sequence_length, batch_size, embedding_dim]
        x = x.transpose(0, 1)  # [num_tokens, batch_size, embedding_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, num_tokens, embedding_dim]
        # Mean pooling over the token sequence
        x = torch.mean(x, dim=1)  # [batch_size, embedding_dim]
        out = self.output_layer(x)
        return out

# ---------------------------
# Main Model Class
# ---------------------------

class Model(nn.Module):
    """
    The main Model class that integrates per-feature embedding modules with a backbone network.
    
    It constructs embedding modules for each numerical feature based on the configuration.
    The supported embedding types are:
       - "baseline" (simple linear mapping)
       - "ple" or "piecewise_linear_encoding" (piecewise linear encoding using precomputed bin boundaries)
       - "periodic" (periodic embedding using sin(k * x + c))
    
    The backbone network is chosen from:
       - MLPBackbone (for MLP-style models)
       - ResNetBackbone (for ResNet-style models)
       - TransformerBackbone (for Transformer-style models)
    
    Methods:
      - build_embeddings(x: Tensor) -> Tensor:
            Processes raw input tensor (shape [batch_size, num_features])
            through per-feature embeddings and aggregates the outputs.
      - forward(x: Tensor) -> Tensor:
            Passes the aggregated embeddings into the backbone and returns predictions.
    
    Args:
      config (dict): Configuration dictionary (from config.yaml).
      num_features (int): Number of numerical features.
      bin_boundaries (list): Optional. A list of bin boundary arrays (one per feature) for PLE.
                             Each element should be convertible to a torch.Tensor.
                             If not provided, default boundaries will be used.
    """
    def __init__(self, config: dict, num_features: int, bin_boundaries: list = None) -> None:
        super(Model, self).__init__()
        self.config = config
        self.num_features = num_features

        # Determine which embedding type to use; default is "baseline".
        # The selected type can be set via config["embeddings"].get("selected_type", ...).
        self.selected_embedding_type = config.get("embeddings", {}).get("selected_type", "baseline").lower()
        # Use output dimension from baseline config as the embedding dimension.
        self.embedding_output_dim = int(config.get("embeddings", {}).get("baseline", {}).get("output_dim", 64))
        
        # Create per-feature embedding modules and register them in a ModuleList.
        self.embeddings = nn.ModuleList()
        for i in range(num_features):
            if self.selected_embedding_type == "baseline":
                embedding_module = BaselineEmbedding(output_dim=self.embedding_output_dim)
            elif self.selected_embedding_type in ["ple", "piecewise_linear_encoding"]:
                # Obtain bin boundaries for this feature if provided; otherwise, use default.
                if bin_boundaries is not None and i < len(bin_boundaries):
                    bins_tensor = torch.tensor(bin_boundaries[i], dtype=torch.float32)
                else:
                    bins_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
                embedding_module = PiecewiseLinearEmbedding(
                    output_dim=self.embedding_output_dim,
                    bin_boundaries=bins_tensor
                )
            elif self.selected_embedding_type == "periodic":
                periodic_cfg = config.get("embeddings", {}).get("periodic", {})
                k_value = int(periodic_cfg.get("k", 16))
                sigma_val = float(periodic_cfg.get("sigma", 0.1))
                embedding_module = PeriodicEmbedding(
                    output_dim=self.embedding_output_dim,
                    k=k_value,
                    sigma=sigma_val
                )
            else:
                # Fallback to baseline
                embedding_module = BaselineEmbedding(output_dim=self.embedding_output_dim)
            self.embeddings.append(embedding_module)

        # Determine the backbone type and instantiate the corresponding module.
        self.backbone_type = config.get("backbone", {}).get("type", "MLP").lower()
        backbone_config = config.get("backbone", {})
        # For MLP and ResNet, the input will be a flattened embedding vector.
        if self.backbone_type in ["mlp", "resnet"]:
            input_dim = num_features * self.embedding_output_dim
        elif self.backbone_type == "transformer":
            # For Transformer, embeddings remain as a sequence.
            input_dim = None  # Not directly used.
        else:
            input_dim = num_features * self.embedding_output_dim

        # Obtain the backbone output dimension if specified; default is 1.
        backbone_output_dim = backbone_config.get("output_dim", 1)

        if self.backbone_type == "mlp":
            self.backbone = MLPBackbone(input_dim=input_dim, output_dim=backbone_output_dim, config=backbone_config)
        elif self.backbone_type == "resnet":
            self.backbone = ResNetBackbone(input_dim=input_dim, output_dim=backbone_output_dim, config=backbone_config)
        elif self.backbone_type == "transformer":
            # For Transformer, provide number of tokens and embedding dimension.
            self.backbone = TransformerBackbone(
                num_tokens=num_features,
                embedding_dim=self.embedding_output_dim,
                output_dim=backbone_output_dim,
                config=backbone_config
            )
        else:
            # Fallback to MLP backbone
            self.backbone = MLPBackbone(input_dim=input_dim, output_dim=backbone_output_dim, config=backbone_config)

    def build_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build aggregated embeddings for input numerical features.

        Args:
          x (torch.Tensor): Input tensor of shape [batch_size, num_features].

        Returns:
          torch.Tensor:
            - For MLP/ResNet: shape [batch_size, num_features * embedding_output_dim].
            - For Transformer: shape [batch_size, num_features, embedding_output_dim].
        """
        embedded_features = []
        # Process each numerical feature individually.
        for i, embedding_module in enumerate(self.embeddings):
            # x[:, i] has shape [batch_size]
            feature_embedding = embedding_module(x[:, i])
            embedded_features.append(feature_embedding)
        if self.backbone_type in ["mlp", "resnet"]:
            # Concatenate embeddings along feature dimension.
            aggregated = torch.cat(embedded_features, dim=1)
        elif self.backbone_type == "transformer":
            # Stack embeddings as a sequence (token dimension).
            aggregated = torch.stack(embedded_features, dim=1)
        else:
            aggregated = torch.cat(embedded_features, dim=1)
        return aggregated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
          x (torch.Tensor): Input tensor of shape [batch_size, num_features].

        Returns:
          torch.Tensor: Model predictions.
        """
        embeddings = self.build_embeddings(x)
        output = self.backbone(embeddings)
        return output
