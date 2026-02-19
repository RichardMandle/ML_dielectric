"""
* VAE architecture
* classifier architecture
* temp pred architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, ModuleList
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, GATConv, GatedGraphConv


import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, LayerNorm, BatchNorm1d
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, JumpingKnowledge, TransformerConv


class VAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim1=512, h_dim2=256, z_dim=128):
        """
        variational autoencoder model architecture

        args:
            x_dim: 
            h_dim1:
            h_dim2:
            z_dim: latent vector dimensions

        """
        super(VAE, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x, x_dim):
        mu, log_var = self.encoder(x.view(-1, x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

class Classifier(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.

    This model is designed to support hyperparameter tuning, including the 
    number of convolutional layers, hidden dimensions, and dropout rates.

    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer (e.g., GCNConv).
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        conv_type=GATConv,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.dropout = dropout
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim

        # Input block
        self.input_block = torch.nn.Sequential(
            conv_type(x_dim, h_dim),
            self.activation
        )

        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    conv_type(h_dim, h_dim),
                    self.activation
                )
                for _ in range(n_conv_blocks)
            ]
        )

        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # Input block
        x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)

        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block[0](x, edge_index)
            x = conv_block[1](x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)

        x = torch.sigmoid(x)
        return x
    

class EnhancedClassifier(torch.nn.Module):
    """
    Enhanced classifier for predicting molecular properties using graph data with multi-head attention.
    
    Features:
    - Multi-head attention with GATConv or GATv2Conv
    - Skip connections
    - Layer normalization
    - Jumping Knowledge connections
    - Optional graph readout methods
    - Dropout scheduling
    - MLP prediction head with batch normalization
    
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer.
        activation (torch.nn.Module): Activation function class.
        pooling_methods (list): List of pooling methods to use.
        use_edge_attr (bool): Whether to use edge attributes.
        edge_dim (int): Dimension of edge features if used.
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        heads=4,
        dropout=0.3,
        conv_type=GATv2Conv,  # GATv2Conv has dynamic attention
        activation=ReLU,
        pooling_methods=["mean", "max", "sum"],
        use_edge_attr=False,
        edge_dim=None,
        jk_mode="cat",  # Options: "cat", "max", "lstm"
    ):
        super(EnhancedClassifier, self).__init__()
        
        self.dropout = dropout
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        self.heads = heads
        self.activation = activation()
        self.use_edge_attr = use_edge_attr
        
        # Define pooling methods
        self.pooling_methods = []
        self.pooling_output_dim = 0
        
        if "mean" in pooling_methods:
            self.pooling_methods.append(("mean", global_mean_pool))
            self.pooling_output_dim += h_dim
        if "max" in pooling_methods:
            self.pooling_methods.append(("max", global_max_pool))  # Use global_max_pool instead of torch.max
            self.pooling_output_dim += h_dim
        if "sum" in pooling_methods:
            self.pooling_methods.append(("sum", global_add_pool))
            self.pooling_output_dim += h_dim
        
        if not self.pooling_methods:
            self.pooling_methods.append(("mean", global_mean_pool))
            self.pooling_output_dim = h_dim
        
        # Input block - multi-head attention (combining all heads)
        if use_edge_attr and edge_dim is not None:
            self.input_conv = conv_type(
                x_dim, h_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim
            )
        else:
            self.input_conv = conv_type(
                x_dim, h_dim // heads, heads=heads, dropout=dropout
            )
        
        self.input_act = self.activation
        self.input_norm = LayerNorm(h_dim)
        
        # Convolutional blocks with skip connections
        self.conv_layers = ModuleList()
        self.norm_layers = ModuleList()
        self.skip_layers = ModuleList()
        
        for i in range(n_conv_blocks):
            if use_edge_attr and edge_dim is not None:
                conv = conv_type(
                    h_dim, h_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim
                )
            else:
                conv = conv_type(
                    h_dim, h_dim // heads, heads=heads, dropout=dropout
                )
            
            self.conv_layers.append(conv)
            self.norm_layers.append(LayerNorm(h_dim))
            
            # Skip connection
            self.skip_layers.append(Linear(h_dim, h_dim))
        
        # Jumping Knowledge connection to combine features from different layers
        self.jk = JumpingKnowledge(mode=jk_mode, channels=h_dim, num_layers=n_conv_blocks+1)
        
        # Output dimension after Jumping Knowledge
        if jk_mode == "cat":
            jk_out_dim = h_dim * (n_conv_blocks + 1)  # +1 for input layer
        else:
            jk_out_dim = h_dim
        
        # MLP prediction head with batch normalization
        mlp_in_dim = jk_out_dim if len(pooling_methods) <= 1 else len(self.pooling_methods) * jk_out_dim
        
        self.mlp = torch.nn.Sequential(
            Linear(mlp_in_dim, 128),
            BatchNorm1d(128),
            self.activation,
            torch.nn.Dropout(dropout),
            Linear(128, 64),
            BatchNorm1d(64),
            self.activation,
            torch.nn.Dropout(dropout * 0.8),  # Gradually reduce dropout
            Linear(64, 32),
            BatchNorm1d(32),
            self.activation,
            torch.nn.Dropout(dropout * 0.6),  # Gradually reduce dropout
            Linear(32, 1)
        )
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Store all layer representations for Jumping Knowledge
        all_layer_outputs = []
        
        # Input block
        if self.use_edge_attr and edge_attr is not None:
            x = self.input_conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.input_conv(x, edge_index)
            
        x = self.input_act(x)
        x = self.input_norm(x)
        all_layer_outputs.append(x)
        
        # Convolutional blocks with skip connections
        for i in range(self.n_conv_blocks):
            identity = x
            
            # Apply attention conv with potential edge attributes
            if self.use_edge_attr and edge_attr is not None:
                conv_out = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            else:
                conv_out = self.conv_layers[i](x, edge_index)
            
            # Skip connection
            skip_out = self.skip_layers[i](identity)
            
            # Combine and normalize
            x = conv_out + skip_out
            x = self.activation(x)
            x = self.norm_layers[i](x)
            
            # Apply dropout with decreasing schedule based on layer depth
            dropout_factor = 1.0 - (i / (2 * self.n_conv_blocks))
            x = F.dropout(x, p=self.dropout * dropout_factor, training=self.training)
            
            all_layer_outputs.append(x)
        
        # Apply Jumping Knowledge
        x = self.jk(all_layer_outputs)
        
        # Multiple pooling methods for better graph representation
        if len(self.pooling_methods) > 1:
            pooled_features = []
            for pool_name, pool_method in self.pooling_methods:
                # All pooling methods now have consistent interfaces
                pooled = pool_method(x, batch)
                pooled_features.append(pooled)
            
            # Concatenate all pooled features
            x = torch.cat(pooled_features, dim=1)
        else:
            # Single pooling method
            _, pool_method = self.pooling_methods[0]
            x = pool_method(x, batch)
        
        # Apply MLP prediction head
        x = self.mlp(x)
        
        # Output probability
        return torch.sigmoid(x)

class GatedGraphClassifier(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.

    This model is designed to support hyperparameter tuning, including the 
    number of convolutional layers, hidden dimensions, and dropout rates.

    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        num_layers (int): Number of layers to consider per GatedGraphConv operation.
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        num_layers=2,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GatedGraphClassifier, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim

        # Input projection layer (needed as GatedGraphConv requires same input and output dims)
        self.input_proj = Linear(x_dim, h_dim)
        
        # Input block - GatedGraphConv requires same input and output dims
        self.input_block = GatedGraphConv(h_dim, num_layers)
        
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                GatedGraphConv(h_dim, num_layers)
                for _ in range(n_conv_blocks)
            ]
        )

        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # Project input features to hidden dimension
        x = self.input_proj(x)
        
        # Input block
        x = self.input_block(x, edge_index)
        x = self.activation(x)

        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x, edge_index)
            x = self.activation(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)

        x = torch.sigmoid(x)
        return x

class TransformerClassifier(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph transformer architecture.

    This model uses TransformerConv layers from PyTorch Geometric to implement
    a graph transformer for molecular property prediction. It supports hyperparameter
    tuning, including the number of transformer layers, hidden dimensions, and dropout rates.

    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the transformer layers.
        n_conv_blocks (int): Number of transformer blocks.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads in transformer layers.
        edge_dim (int, optional): Edge feature dimension. Set to None if no edge features.
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        heads=4,
        edge_dim=None,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(TransformerClassifier, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        self.heads = heads

        # Input block - TransformerConv from input dimension to hidden dimension
        self.input_block = torch.nn.Sequential(
            TransformerConv(
                in_channels=x_dim, 
                out_channels=h_dim // heads,  # Divide by heads to keep total dim consistent
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True  # Enable learnable skip connection
            ),
            self.activation
        )

        # Transformer blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    TransformerConv(
                        in_channels=h_dim, 
                        out_channels=h_dim // heads,  # Divide by heads to keep total dim consistent
                        heads=heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True  # Enable learnable skip connection
                    ),
                    self.activation
                )
                for _ in range(n_conv_blocks)
            ]
        )

        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Input block
        if edge_attr is not None:
            x = self.input_block[0](x, edge_index, edge_attr=edge_attr)
        else:
            x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)

        # Transformer blocks
        for conv_block in self.conv_blocks:
            if edge_attr is not None:
                x_new = conv_block[0](x, edge_index, edge_attr=edge_attr)
            else:
                x_new = conv_block[0](x, edge_index)
            x_new = conv_block[1](x_new)
            x = F.dropout(x_new, p=self.dropout, training=self.training)

        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        x = torch.sigmoid(x)
        return x

class GINClassifier(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.
    This model is designed to support hyperparameter tuning, including the
    number of convolutional layers, hidden dimensions, and dropout rates.
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer (e.g., GINConv).
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        conv_type=GINConv,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GINClassifier, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        
        # Create MLP for GINConv input layer
        input_nn = Sequential(
            Linear(x_dim, h_dim),
            self.activation,
            Linear(h_dim, h_dim)
        )
        
        # Input block
        self.input_block = torch.nn.Sequential(
            conv_type(input_nn),
            self.activation
        )
        
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    conv_type(Sequential(
                        Linear(h_dim, h_dim),
                        self.activation,
                        Linear(h_dim, h_dim)
                    )),
                    self.activation
                )
                for _ in range(n_conv_blocks)
            ]
        )
        
        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)
    
    def forward(self, x, edge_index, batch):
        # Input block
        x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)
        
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block[0](x, edge_index)
            x = conv_block[1](x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        
        return x
    
class GCNClassifier(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.
    This model is designed to support hyperparameter tuning, including the
    number of convolutional layers, hidden dimensions, and dropout rates.
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer (e.g., GCNConv).
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        conv_type=GCNConv,  # Changed from GATConv to GCNConv
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GCNClassifier, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.dropout = dropout
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        # Input block
        self.input_block = torch.nn.Sequential(
            conv_type(x_dim, h_dim),
            self.activation
        )
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    conv_type(h_dim, h_dim),
                    self.activation
                )
                for _ in range(n_conv_blocks)  # Fixed asterisk to underscore
            ]
        )
        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)
        
    def forward(self, x, edge_index, batch):
        # Input block
        x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block[0](x, edge_index)
            x = conv_block[1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x

class Predictor(torch.nn.Module):
    """
    Graph Attention Network (GAT) for predicting temperature from graph data.

    Args:
        x_dim (int): Input feature dimension.
        h_dim (int): Hidden layer dimension.
        n_layers (int): Number of GATConv layers.
        dropout (float): Dropout probability.
        activation (callable): Activation function.
        pooling (callable): Global pooling function (default: global_add_pool).
    """
    def __init__(self, x_dim=79, h_dim=64, n_conv_blocks=3, dropout=0.5, activation=ReLU, pooling=global_add_pool):
        super(Predictor, self).__init__()

        self.h_dim = h_dim
        self.n_layers = n_conv_blocks
        self.dropout = dropout
        self.activation = activation()
        self.pooling = pooling
        gat_heads = 8

        # Build GAT layers
        self.gat_layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            input_dim = x_dim if i == 0 else h_dim * gat_heads
            gat_layer = GATConv(input_dim, h_dim, heads=gat_heads, concat=True)
            self.gat_layers.append(gat_layer)

        # Fully connected layers
        self.lin1 = Linear(h_dim * gat_heads, h_dim)
        self.lin2 = Linear(h_dim, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings through GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = self.activation(x)

        # Pooling
        x = self.pooling(x, batch)

        # Fully connected layers
        x = self.lin1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
    
class EnhancedPredictor(torch.nn.Module):
    """
    Enhanced classifier for predicting molecular properties using graph data with multi-head attention.
    
    Features:
    - Multi-head attention with GATConv or GATv2Conv
    - Skip connections
    - Layer normalization
    - Jumping Knowledge connections
    - Optional graph readout methods
    - Dropout scheduling
    - MLP prediction head with batch normalization
    
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer.
        activation (torch.nn.Module): Activation function class.
        pooling_methods (list): List of pooling methods to use.
        use_edge_attr (bool): Whether to use edge attributes.
        edge_dim (int): Dimension of edge features if used.
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        heads=4,
        dropout=0.3,
        conv_type=GATv2Conv,  # GATv2Conv has dynamic attention
        activation=ReLU,
        pooling_methods=["mean", "max", "sum"],
        use_edge_attr=False,
        edge_dim=None,
        jk_mode="cat",  # Options: "cat", "max", "lstm"
    ):
        super(EnhancedPredictor, self).__init__()
        
        self.dropout = dropout
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        self.heads = heads
        self.activation = activation()
        self.use_edge_attr = use_edge_attr
        
        # Define pooling methods
        self.pooling_methods = []
        self.pooling_output_dim = 0
        
        if "mean" in pooling_methods:
            self.pooling_methods.append(("mean", global_mean_pool))
            self.pooling_output_dim += h_dim
        if "max" in pooling_methods:
            self.pooling_methods.append(("max", global_max_pool))  # Use global_max_pool instead of torch.max
            self.pooling_output_dim += h_dim
        if "sum" in pooling_methods:
            self.pooling_methods.append(("sum", global_add_pool))
            self.pooling_output_dim += h_dim
        
        if not self.pooling_methods:
            self.pooling_methods.append(("mean", global_mean_pool))
            self.pooling_output_dim = h_dim
        
        # Input block - multi-head attention (combining all heads)
        if use_edge_attr and edge_dim is not None:
            self.input_conv = conv_type(
                x_dim, h_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim
            )
        else:
            self.input_conv = conv_type(
                x_dim, h_dim // heads, heads=heads, dropout=dropout
            )
        
        self.input_act = self.activation
        self.input_norm = LayerNorm(h_dim)
        
        # Convolutional blocks with skip connections
        self.conv_layers = ModuleList()
        self.norm_layers = ModuleList()
        self.skip_layers = ModuleList()
        
        for i in range(n_conv_blocks):
            if use_edge_attr and edge_dim is not None:
                conv = conv_type(
                    h_dim, h_dim // heads, heads=heads, dropout=dropout, edge_dim=edge_dim
                )
            else:
                conv = conv_type(
                    h_dim, h_dim // heads, heads=heads, dropout=dropout
                )
            
            self.conv_layers.append(conv)
            self.norm_layers.append(LayerNorm(h_dim))
            
            # Skip connection
            self.skip_layers.append(Linear(h_dim, h_dim))
        
        # Jumping Knowledge connection to combine features from different layers
        self.jk = JumpingKnowledge(mode=jk_mode, channels=h_dim, num_layers=n_conv_blocks+1)
        
        # Output dimension after Jumping Knowledge
        if jk_mode == "cat":
            jk_out_dim = h_dim * (n_conv_blocks + 1)  # +1 for input layer
        else:
            jk_out_dim = h_dim
        
        # MLP prediction head with batch normalization
        mlp_in_dim = jk_out_dim if len(pooling_methods) <= 1 else len(self.pooling_methods) * jk_out_dim
        
        self.mlp = torch.nn.Sequential(
            Linear(mlp_in_dim, 128),
            LayerNorm(128),  # <-- Works with any batch size
            self.activation,
            torch.nn.Dropout(dropout),
            Linear(128, 64),
            LayerNorm(64),
            self.activation,
            torch.nn.Dropout(dropout * 0.8),
            Linear(64, 32),
            LayerNorm(32),
            self.activation,
            torch.nn.Dropout(dropout * 0.6),
            Linear(32, 1)
        )
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Store all layer representations for Jumping Knowledge
        all_layer_outputs = []
        
        # Input block
        if self.use_edge_attr and edge_attr is not None:
            x = self.input_conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.input_conv(x, edge_index)
            
        x = self.input_act(x)
        x = self.input_norm(x)
        all_layer_outputs.append(x)
        
        # Convolutional blocks with skip connections
        for i in range(self.n_conv_blocks):
            identity = x
            
            # Apply attention conv with potential edge attributes
            if self.use_edge_attr and edge_attr is not None:
                conv_out = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            else:
                conv_out = self.conv_layers[i](x, edge_index)
            
            # Skip connection
            skip_out = self.skip_layers[i](identity)
            
            # Combine and normalize
            x = conv_out + skip_out
            x = self.activation(x)
            x = self.norm_layers[i](x)
            
            # Apply dropout with decreasing schedule based on layer depth
            dropout_factor = 1.0 - (i / (2 * self.n_conv_blocks))
            x = F.dropout(x, p=self.dropout * dropout_factor, training=self.training)
            
            all_layer_outputs.append(x)
        
        # Apply Jumping Knowledge
        x = self.jk(all_layer_outputs)
        
        # Multiple pooling methods for better graph representation
        if len(self.pooling_methods) > 1:
            pooled_features = []
            for pool_name, pool_method in self.pooling_methods:
                # All pooling methods now have consistent interfaces
                pooled = pool_method(x, batch)
                pooled_features.append(pooled)
            
            # Concatenate all pooled features
            x = torch.cat(pooled_features, dim=1)
        else:
            # Single pooling method
            _, pool_method = self.pooling_methods[0]
            x = pool_method(x, batch)
        
        # Apply MLP prediction head
        x = self.mlp(x)
        
        # Output probability
        return x
    
class TransformerPredictor(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph transformer architecture.

    This model uses TransformerConv layers from PyTorch Geometric to implement
    a graph transformer for molecular property prediction. It supports hyperparameter
    tuning, including the number of transformer layers, hidden dimensions, and dropout rates.

    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the transformer layers.
        n_conv_blocks (int): Number of transformer blocks.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads in transformer layers.
        edge_dim (int, optional): Edge feature dimension. Set to None if no edge features.
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        heads=4,
        edge_dim=None,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(TransformerPredictor, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        self.heads = heads

        # Input block - TransformerConv from input dimension to hidden dimension
        self.input_block = torch.nn.Sequential(
            TransformerConv(
                in_channels=x_dim, 
                out_channels=h_dim // heads,  # Divide by heads to keep total dim consistent
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True  # Enable learnable skip connection
            ),
            self.activation
        )

        # Transformer blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    TransformerConv(
                        in_channels=h_dim, 
                        out_channels=h_dim // heads,  # Divide by heads to keep total dim consistent
                        heads=heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True  # Enable learnable skip connection
                    ),
                    self.activation
                )
                for _ in range(n_conv_blocks)
            ]
        )

        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Input block
        if edge_attr is not None:
            x = self.input_block[0](x, edge_index, edge_attr=edge_attr)
        else:
            x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)

        # Transformer blocks
        for conv_block in self.conv_blocks:
            if edge_attr is not None:
                x_new = conv_block[0](x, edge_index, edge_attr=edge_attr)
            else:
                x_new = conv_block[0](x, edge_index)
            x_new = conv_block[1](x_new)
            x = F.dropout(x_new, p=self.dropout, training=self.training)

        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x

class GatedGraphPredictor(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.

    This model is designed to support hyperparameter tuning, including the 
    number of convolutional layers, hidden dimensions, and dropout rates.

    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        num_layers (int): Number of layers to consider per GatedGraphConv operation.
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        num_layers=2,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GatedGraphPredictor, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim

        # Input projection layer (needed as GatedGraphConv requires same input and output dims)
        self.input_proj = Linear(x_dim, h_dim)
        
        # Input block - GatedGraphConv requires same input and output dims
        self.input_block = GatedGraphConv(h_dim, num_layers)
        
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                GatedGraphConv(h_dim, num_layers)
                for _ in range(n_conv_blocks)
            ]
        )

        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # Project input features to hidden dimension
        x = self.input_proj(x)
        
        # Input block
        x = self.input_block(x, edge_index)
        x = self.activation(x)

        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x, edge_index)
            x = self.activation(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x
    
class GCNPredictor(torch.nn.Module):
    """
    Classifier for predicting molecular properties using graph data.
    This model is designed to support hyperparameter tuning, including the
    number of convolutional layers, hidden dimensions, and dropout rates.
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer (e.g., GCNConv).
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        conv_type=GCNConv,  # Changed from GATConv to GCNConv
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GCNPredictor, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.dropout = dropout
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        # Input block
        self.input_block = torch.nn.Sequential(
            conv_type(x_dim, h_dim),
            self.activation
        )
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    conv_type(h_dim, h_dim),
                    self.activation
                )
                for _ in range(n_conv_blocks)  # Fixed asterisk to underscore
            ]
        )
        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)
        
    def forward(self, x, edge_index, batch):
        # Input block
        x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block[0](x, edge_index)
            x = conv_block[1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
class GINPredictor(torch.nn.Module):
    """
    Predictor for predicting molecular properties using graph data.
    This model is designed to support hyperparameter tuning, including the
    number of convolutional layers, hidden dimensions, and dropout rates.
    Args:
        x_dim (int): Dimension of input node features.
        h_dim (int): Number of hidden units in the convolutional layers.
        n_conv_blocks (int): Number of graph convolutional blocks.
        dropout (float): Dropout probability.
        conv_type (class): Type of graph convolutional layer (e.g., GINConv).
        activation (torch.nn.Module): Activation function (default: ReLU).
        pooling (callable): Global pooling function (default: global_mean_pool).
    """
    def __init__(
        self,
        x_dim=79,
        h_dim=64,
        n_conv_blocks=3,
        dropout=0.5,
        conv_type=GINConv,
        activation=ReLU,
        pooling=global_mean_pool,
    ):
        super(GINPredictor, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation()
        self.n_conv_blocks = n_conv_blocks
        self.h_dim = h_dim
        
        # Create MLP for GINConv input layer
        input_nn = Sequential(
            Linear(x_dim, h_dim),
            self.activation,
            Linear(h_dim, h_dim)
        )
        
        # Input block
        self.input_block = torch.nn.Sequential(
            conv_type(input_nn),
            self.activation
        )
        
        # Convolutional blocks
        self.conv_blocks = ModuleList(
            [
                torch.nn.Sequential(
                    conv_type(Sequential(
                        Linear(h_dim, h_dim),
                        self.activation,
                        Linear(h_dim, h_dim)
                    )),
                    self.activation
                )
                for _ in range(n_conv_blocks)
            ]
        )
        
        # Fully connected layers
        self.linear = Linear(h_dim, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 1)
    
    def forward(self, x, edge_index, batch):
        # Input block
        x = self.input_block[0](x, edge_index)
        x = self.input_block[1](x)
        
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block[0](x, edge_index)
            x = conv_block[1](x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling and dropout
        x = self.pooling(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fully connected layers with sigmoid activation
        x = self.linear(x)
        x = self.linear2(x)
        x = self.linear3(x)
        
        return x