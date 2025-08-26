import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import List, Dict, Optional, Tuple, Union
from model.attention.base_robust_method import BaseRobustMethod

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c7685960.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class EnhancedVGG(nn.Module):
    """
    Enhanced VGG with Batch Normalization and improved architecture.
    Based on custom VGG16 structure with additional normalization and dropout.
    """

    def __init__(self, features: nn.Module, num_classes: int, input_channels: int = 3,
                 pretrained: bool = False, robust_method: Optional[BaseRobustMethod] = None,
                 depth: int = 16, dropout_rate: float = 0.5):
        super(EnhancedVGG, self).__init__()
        self.features = features
        self.depth = depth
        self.dropout_rate = dropout_rate

        # Enhanced classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

        if pretrained:
            self.load_pretrained_weights(depth)

        self.robust_method = robust_method

        # Adjust the first layer for custom input channels
        if input_channels != 3:
            self.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=3, padding=1)
            # Re-initialize the modified first layer
            nn.init.kaiming_normal_(
                self.features[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        if self.robust_method:
            # Flatten for robust method
            x_flat = x.view(x.size(0), -1)
            x_flat, _ = self.robust_method(x_flat, x_flat, x_flat)
            return x_flat  # Return flattened output for robust method

        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize model weights using appropriate initialization schemes."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, depth: int):
        """Load pretrained weights from torchvision models."""
        url = model_urls.get(f'vgg{depth}')
        if url is None:
            print(
                f"Warning: No pretrained model available for VGG{depth}, using random initialization")
            return

        try:
            pretrained_dict = load_state_dict_from_url(url, progress=True)
            model_dict = self.state_dict()

            # Filter out classifier weights since we have a different classifier structure
            # Only load feature weights that match
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and 'classifier' not in k and 'features' in k:
                    # Check if dimensions match (important for batch norm layers)
                    if model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(
                            f"Skipping {k} due to shape mismatch: {model_dict[k].shape} vs {v.shape}")

            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"Loaded pretrained weights for VGG{depth} features")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")


def make_enhanced_vgg_features(cfg: List[Union[int, str]], input_channels: int = 3) -> nn.Sequential:
    """
    Create enhanced VGG feature layers with batch normalization.

    Args:
        cfg: Configuration list defining the architecture
        input_channels: Number of input channels

    Returns:
        nn.Sequential: Feature extraction layers
    """
    layers = []
    in_channels = input_channels

    for i, v in enumerate(cfg):
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([
                conv2d,
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True)
            ])
            in_channels = v

    return nn.Sequential(*layers)


def get_enhanced_vgg(depth: int, pretrained: bool = False, input_channels: int = 3,
                     num_classes: int = 1000, robust_method: Optional[BaseRobustMethod] = None,
                     dropout_rate: float = 0.5) -> EnhancedVGG:
    """
    Get Enhanced VGG model with specified depth and configuration.

    Args:
        depth: VGG depth (11, 13, 16, 19)
        pretrained: Whether to load pretrained weights
        input_channels: Number of input channels
        num_classes: Number of output classes
        robust_method: Optional robust method to apply
        dropout_rate: Dropout rate for classifier layers

    Returns:
        EnhancedVGG: The model instance
    """
    # Enhanced configurations with proper VGG structure
    depth_to_cfg = {
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    if depth not in depth_to_cfg:
        raise ValueError(
            f"Unsupported Enhanced VGG depth: {depth}. Supported depths: {list(depth_to_cfg.keys())}")

    cfg = depth_to_cfg[depth]
    features = make_enhanced_vgg_features(cfg, input_channels)

    model = EnhancedVGG(
        features=features,
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained,
        robust_method=robust_method,
        depth=depth,
        dropout_rate=dropout_rate
    )

    return model

# Alias for compatibility with existing model loader


def get_vgg_enhanced(depth: int, pretrained: bool = False, input_channels: int = 3,
                     num_classes: int = 1000, robust_method: Optional[BaseRobustMethod] = None) -> EnhancedVGG:
    """
    Alias for get_enhanced_vgg for compatibility with model loader.
    """
    return get_enhanced_vgg(depth, pretrained, input_channels, num_classes, robust_method)

# Main function for VGGNet Enhanced


def get_vggnet(depth: int, pretrained: bool = False, input_channels: int = 3,
               num_classes: int = 1000, robust_method: Optional[BaseRobustMethod] = None) -> EnhancedVGG:
    """
    Get Enhanced VGG model for classification tasks.
    """
    return get_enhanced_vgg(depth, pretrained, input_channels, num_classes, robust_method)
