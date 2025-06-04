import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism to learn frame importance."""
    
    def __init__(self, feature_dim, num_heads=4, dropout=0.1, max_seq_length=30):
        """
        Initialize the temporal attention module.
        
        Args:
            feature_dim: Dimension of the input features
            num_heads: Number of attention heads
            dropout: Dropout rate for attention
            max_seq_length: Maximum number of frames to support
        """
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Positional encoding - learnable parameter
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, max_seq_length, feature_dim)
        )
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Forward pass of the temporal attention module.
        
        Args:
            x: Input tensor of shape [batch_size, frames, feature_dim]
            
        Returns:
            pooled: Aggregated features of shape [batch_size, feature_dim]
            attn_weights: Attention weights for visualization
        """
        # Apply layer normalization
        x = self.norm(x)
        
        # Add positional encoding
        seq_length = x.size(1)
        x = x + self.pos_encoder[:, :seq_length, :]
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x
        )
        
        # Weighted average based on attention scores
        pooled = attn_output.mean(dim=1)
        
        return pooled, attn_weights


class TemporalConvolution(nn.Module):
    """Temporal convolution module to learn patterns across frames."""
    
    def __init__(self, feature_dim, kernel_size=3):
        """
        Initialize the temporal convolution module.
        
        Args:
            feature_dim: Dimension of the input features
            kernel_size: Size of the convolutional kernel
        """
        super(TemporalConvolution, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        """
        Forward pass of the temporal convolution module.
        
        Args:
            x: Input tensor of shape [batch_size, feature_dim, frames]
            
        Returns:
            pooled: Aggregated features of shape [batch_size, feature_dim]
        """
        pooled = self.conv_block(x).squeeze(-1)
        return pooled, None  # Return None for attn_weights to maintain API consistency


class AdaptivePooling(nn.Module):
    """Simple adaptive pooling module."""
    
    def __init__(self, feature_dim):
        """
        Initialize the adaptive pooling module.
        
        Args:
            feature_dim: Dimension of the input features
        """
        super(AdaptivePooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Forward pass of the adaptive pooling module.
        
        Args:
            x: Input tensor of shape [batch_size, feature_dim, frames]
            
        Returns:
            pooled: Aggregated features of shape [batch_size, feature_dim]
        """
        pooled = self.pool(x).squeeze(-1)
        return pooled, None  # Return None for attn_weights to maintain API consistency


class TemporalRNN(nn.Module):
    """RNN-based temporal module to learn sequential patterns across frames."""
    
    def __init__(self, feature_dim, hidden_dim=512, rnn_type='lstm', num_layers=2, bidirectional=True, dropout=0.2):
        """
        Initialize the temporal RNN module.
        
        Args:
            feature_dim: Dimension of the input features
            hidden_dim: Dimension of the RNN hidden state
            rnn_type: Type of RNN to use ('rnn', 'lstm', or 'gru')
            num_layers: Number of recurrent layers
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout rate for RNN layers
        """
        super(TemporalRNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Choose RNN type
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Projection layer to transform hidden state to feature dimension if needed
        output_dim = hidden_dim * self.num_directions
        if output_dim != feature_dim:
            self.projection = nn.Linear(output_dim, feature_dim)
        else:
            self.projection = nn.Identity()
            
        # Normalization for stability
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Forward pass of the temporal RNN module.
        
        Args:
            x: Input tensor of shape [batch_size, frames, feature_dim]
            
        Returns:
            pooled: Aggregated features of shape [batch_size, feature_dim]
            attention_weights: None (for consistency with other modules)
        """
        # Apply normalization
        x = self.norm(x)
        
        # Pass through RNN
        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(x)
        else:
            output, hidden = self.rnn(x)
        
        # Get final hidden states (from both directions if bidirectional)
        if self.bidirectional:
            # For bidirectional RNNs, hidden has shape [num_layers*num_directions, batch, hidden_dim]
            # We concatenate the last layer's hidden states from both directions
            last_layer_hidden = hidden[self.num_layers*self.num_directions-2:self.num_layers*self.num_directions, :, :]
            last_hidden = last_layer_hidden.transpose(0, 1).contiguous().view(-1, self.hidden_dim * self.num_directions)
        else:
            # For unidirectional RNNs, just take the last layer
            last_hidden = hidden[-1]
        
        # Apply projection layer if needed
        pooled = self.projection(last_hidden)
        
        return pooled, None


class EnhancedFrameCNN(nn.Module):
    """
    Enhanced 2D CNN model that processes each frame individually and then aggregates
    with configurable temporal aggregation methods.
    """
    
    def __init__(self, 
                 base_model='resnet18', 
                 pretrained=True, 
                 dropout_rate=0.5,
                 temporal_mode='attention',  # Options: 'attention', 'convolution', 'pooling', 'rnn', 'lstm', 'gru'
                 attention_heads=4,
                 temporal_kernel_size=3,
                 rnn_hidden_dim=512,
                 rnn_num_layers=2,
                 rnn_bidirectional=True,
                 store_attention_weights=False):
        """
        Initialize the EnhancedFrameCNN model.
        
        Args:
            base_model: Base model architecture ('resnet18', 'resnet50', 'mobilenet_v2', etc.)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for the classifier
            temporal_mode: Method for temporal aggregation 
                          ('attention', 'convolution', 'pooling', 'rnn', 'lstm', 'gru')
            attention_heads: Number of attention heads (if using attention)
            temporal_kernel_size: Kernel size for temporal convolution (if using convolution)
            rnn_hidden_dim: Hidden dimension for RNN/LSTM/GRU (if using recurrent networks)
            rnn_num_layers: Number of recurrent layers (if using recurrent networks)
            rnn_bidirectional: Whether to use bidirectional RNN/LSTM/GRU
            store_attention_weights: Whether to store the last attention weights for visualization
        """
        super(EnhancedFrameCNN, self).__init__()
        
        self.store_attention_weights = store_attention_weights
        self.last_attn_weights = None
        
        # Import base model based on name
        if base_model == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification layer
        elif base_model == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif base_model == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v2(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            self.feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_v2_s(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'efficientnet_v2_m':
            weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_v2_m(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'efficientnet_v2_l':
            weights = EfficientNet_V2_L_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_v2_l(weights=weights)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            self.feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'convnext_base':
            weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_base(weights=weights)
            self.feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif base_model == 'convnext_large':
            weights = ConvNeXt_Large_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_large(weights=weights)
            self.feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Temporal aggregation
        self.temporal_mode = temporal_mode
        
        if temporal_mode == 'attention':
            self.temporal_aggregation = TemporalAttention(
                feature_dim=self.feature_dim,
                num_heads=attention_heads,
                dropout=dropout_rate * 0.5  # Lower dropout for attention
            )
        elif temporal_mode == 'convolution':
            self.temporal_aggregation = TemporalConvolution(
                feature_dim=self.feature_dim,
                kernel_size=temporal_kernel_size
            )
        elif temporal_mode == 'pooling':
            self.temporal_aggregation = AdaptivePooling(
                feature_dim=self.feature_dim
            )
        elif temporal_mode in ['rnn', 'lstm', 'gru']:
            self.temporal_aggregation = TemporalRNN(
                feature_dim=self.feature_dim,
                hidden_dim=rnn_hidden_dim,
                rnn_type=temporal_mode,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                dropout=dropout_rate * 0.5
            )
        else:
            raise ValueError(f"Unsupported temporal mode: {temporal_mode}")
        
        # Enhanced classifier with more regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
            # Remove sigmoid activation when using BCEWithLogitsLoss
        )
        
        print(f"Created {base_model} with {temporal_mode} temporal aggregation")
        print(f"Feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        """
        Forward pass of the EnhancedFrameCNN model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, frames, height, width]
              or [batch_size, frames, height, width, channels]
              
        Returns:
            output: Model predictions
        """
        # Handle both input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # [batch_size, channels, frames, height, width]
                batch_size, channels, num_frames, height, width = x.shape
            else:  # [batch_size, frames, height, width, channels]
                batch_size, num_frames, height, width, channels = x.shape
                x = x.permute(0, 4, 1, 2, 3)  # Convert to [batch_size, channels, frames, height, width]
        else:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")
        
        # For memory efficiency: optionally subsample frames
        frame_subsample = 2  # Use every other frame
        if num_frames > 10:  # Only subsample if we have many frames
            x = x[:, :, ::frame_subsample, :, :]
            num_frames = x.shape[2]
        
        # Reshape for frame-by-frame processing
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, frames, channels, height, width]
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        # Extract features from each frame
        features = self.backbone(x)  # [batch_size * frames, feature_dim]
        
        # Reshape back to separate batch and frames
        features = features.reshape(batch_size, num_frames, -1)  # [batch_size, frames, feature_dim]
        
        # Temporal aggregation
        if self.temporal_mode in ['attention', 'rnn', 'lstm', 'gru']:
            # For attention and RNN-based methods, input is [batch_size, frames, feature_dim]
            pooled, attn_weights = self.temporal_aggregation(features)
        else:
            # For convolution and pooling, reshape to [batch_size, feature_dim, frames]
            features = features.permute(0, 2, 1)
            pooled, attn_weights = self.temporal_aggregation(features)
        
        # Store attention weights if needed
        if self.store_attention_weights and attn_weights is not None:
            self.last_attn_weights = attn_weights.detach()
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_weights(self):
        """
        Get the last computed attention weights for visualization.
        
        Returns:
            Last attention weights or None if not available
        """
        return self.last_attn_weights


# Example usage:
"""
# Create model with different temporal aggregation methods
# Attention-based model
model_attention = EnhancedFrameCNN(
    base_model='resnet18',
    temporal_mode='attention',
    attention_heads=4,
    store_attention_weights=True
)

# LSTM-based model
model_lstm = EnhancedFrameCNN(
    base_model='resnet18',
    temporal_mode='lstm',
    rnn_hidden_dim=512,
    rnn_num_layers=2,
    rnn_bidirectional=True
)

# GRU-based model
model_gru = EnhancedFrameCNN(
    base_model='mobilenet_v2',
    temporal_mode='gru',
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    rnn_bidirectional=False
)

# Simple RNN model
model_rnn = EnhancedFrameCNN(
    base_model='resnet50',
    temporal_mode='rnn',
    rnn_hidden_dim=512,
    rnn_num_layers=2
)

# Convolution-based model
model_conv = EnhancedFrameCNN(
    base_model='resnet18',
    temporal_mode='convolution',
    temporal_kernel_size=3
)

# Basic pooling model
model_pool = EnhancedFrameCNN(
    base_model='resnet18',
    temporal_mode='pooling'
)

# Use with EfficientNet
model_efficient = EnhancedFrameCNN(
    base_model='efficientnet_v2_s',
    temporal_mode='lstm'
)
"""


def visualize_attention(model, video_frames, original_fps=15, output_path="attention_viz.mp4"):
    """
    Visualize attention weights overlaid on video frames.
    
    Args:
        model: Trained EnhancedFrameCNN model with attention
        video_frames: Video frames tensor [1, frames, height, width, channels] or [1, channels, frames, height, width]
        original_fps: Original frames per second of the video
        output_path: Path to save the visualization
    
    Returns:
        Path to the saved visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.colors import Normalize
    import cv2
    import os
    
    if not hasattr(model, 'get_attention_weights') or not model.store_attention_weights:
        raise ValueError("Model does not support attention visualization")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Forward pass to compute attention weights
    with torch.no_grad():
        _ = model(video_frames)
    
    # Get attention weights
    attn_weights = model.get_attention_weights()
    if attn_weights is None:
        raise ValueError("No attention weights available. Make sure to set store_attention_weights=True")
    
    # Extract weights for visualization (average across heads if multi-head)
    if attn_weights.dim() > 3:
        weights = attn_weights.mean(dim=1)  # Average over heads
    else:
        weights = attn_weights
    
    # Get frames in numpy format
    if video_frames.shape[1] == 3:  # [batch, channels, frames, height, width]
        frames_np = video_frames[0].permute(1, 2, 3, 0).cpu().numpy()
    else:  # [batch, frames, height, width, channels]
        frames_np = video_frames[0].cpu().numpy()
    
    # Normalize attention weights for visualization
    weights_np = weights[0].cpu().numpy()
    weights_norm = Normalize()(weights_np.mean(axis=1))  # Use mean if weights are 2D
    
    # Create colormap for attention visualization
    cmap = cm.get_cmap('jet')
    
    # Setup video writer
    height, width = frames_np.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
    
    # Create visualization frames
    for i, frame in enumerate(frames_np):
        if i >= len(weights_norm):
            break
            
        # Convert to RGB uint8 if needed
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
            
        # Create attention heatmap
        attention = cmap(weights_norm[i])[:, :3]  # Remove alpha channel
        attention_map = (attention * 255).astype(np.uint8)
        
        # Resize attention map to match frame
        attention_resized = cv2.resize(attention_map, (width, height))
        
        # Overlay attention on frame (alpha blending)
        alpha = 0.4  # Transparency of the overlay
        overlaid_frame = cv2.addWeighted(
            frame, 1 - alpha, 
            attention_resized, alpha, 
            0
        )
        
        # Add frame to video
        video_writer.write(cv2.cvtColor(overlaid_frame, cv2.COLOR_RGB2BGR))
    
    video_writer.release()
    print(f"Attention visualization saved to {output_path}")
    
    return output_path