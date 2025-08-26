"""
Memory efficient model wrapper.
Compatible implementation for model memory optimization.
"""

import torch
import torch.nn as nn


class MemoryEfficientModel:
    """Memory efficient model wrapper compatible with the existing codebase."""

    def __init__(self, model_builder, device='cuda', fp16=False):
        """
        Initialize memory efficient model.

        Args:
            model_builder: Function that creates the model
            device: Device to load the model on
            fp16: Whether to use half precision
        """
        self.model_builder = model_builder
        self.device = device
        self.fp16 = fp16
        self.model = None

    def load_model(self):
        """Load the model using the builder function."""
        if self.model is None:
            self.model = self.model_builder()
            self.model = self.model.to(self.device)

            if self.fp16:
                self.model = self.model.half()

        return self.model

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        if self.model is None:
            self.model = self.load_model()
        return self.model(*args, **kwargs)

    def train(self, mode=True):
        """Set training mode."""
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self

    def parameters(self):
        """Get model parameters."""
        if self.model is None:
            self.model = self.load_model()
        return self.model.parameters()

    def state_dict(self):
        """Get model state dict."""
        if self.model is None:
            self.model = self.load_model()
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        if self.model is None:
            self.model = self.load_model()
        return self.model.load_state_dict(state_dict)


def make_memory_efficient(model, device='cuda', fp16=False):
    """Simple function to wrap model for memory efficiency."""
    def model_builder():
        return model
    return MemoryEfficientModel(model_builder, device, fp16)
