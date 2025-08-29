"""
Clean, detection-ready YOLOv8 wrapper.

This module provides a lightweight, well-documented wrapper around the
Ultralytics YOLOv8 implementation that focuses on detection. It exposes a
consistent API for loading models, running inference, adapting input
channels, and returning standardized detection outputs (boxes, scores,
and class ids).

Notes:
- Requires the `ultralytics` package. If it's not available this module
  raises a clear ImportError at import time.
- The wrapper intentionally stays small: it does not try to modify the
  internal YOLO architecture (no FasterNet/EMA hacks here). Those
  experimental modifications can be added in separate modules for
  research experiments to keep production code stable.

"""

from typing import List, Optional, Union
import logging

import torch

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    ULTRALYTICS_AVAILABLE = True
except Exception as e:  # ImportError or runtime errors
    ULTRALYTICS_AVAILABLE = False
    logging.warning("ultralytics not available: %s", e)


class YOLOv8Detector:
    """Detection-ready wrapper for Ultralytics YOLOv8.

    Example:
        detector = YOLOv8Detector('small', pretrained=True)
        results = detector.predict(image_tensor_or_path)

    The `predict` method returns a list of dicts, one per image, with keys:
        - 'boxes': Tensor[N,4] in xyxy format
        - 'scores': Tensor[N]
        - 'labels': Tensor[N]
        - 'orig_size': (H, W)
    """

    VARIANT_MAP = {
        'nano': 'yolov8n.pt',
        'small': 'yolov8s.pt',
        'medium': 'yolov8m.pt',
        'large': 'yolov8l.pt',
        'xlarge': 'yolov8x.pt',
        'x': 'yolov8x.pt',
    }

    def __init__(self,
                 variant: str = 'small',
                 pretrained: bool = False,
                 device: Optional[Union[str, torch.device]] = None,
                 input_channels: int = 3):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is required to use YOLOv8Detector. Install with: pip install ultralytics")

        self.variant = variant.lower()
        base = self.variant.split('-')[0]
        if base not in self.VARIANT_MAP:
            raise ValueError(f"Unknown YOLOv8 variant: {variant}")

        model_path = self.VARIANT_MAP[base]

        # Use YOLO() which can load both .pt and .yaml
        if pretrained:
            self.model = YOLO(model_path)
        else:
            # Load model from yaml (ultralytics accepts yaml or model string)
            yaml_path = model_path.replace('.pt', '.yaml')
            try:
                self.model = YOLO(yaml_path)
            except Exception:
                # Fallback: still try the pt path (may raise helpful error)
                self.model = YOLO(model_path)

        # Set device
        if device is not None:
            self.to(device)

        # Adapt first conv if input channels != 3
        if input_channels != 3:
            try:
                self._adapt_input_channels(input_channels)
            except Exception as e:
                logging.warning("Failed to adapt input channels: %s", e)

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        try:
            self.model.to(device)
        except Exception:
            # Some ultralytics wrappers use internal .model (nn.Module)
            if hasattr(self.model, 'model'):
                self.model.model.to(device)
        return self

    def _adapt_input_channels(self, input_channels: int):
        """Try to change the first Conv2d to handle different input channels.

        This will preserve the existing 3-channel weights where possible and
        initialize the extra channels with kaiming normal.
        """
        # Try to find a Conv module in the internal model
        m = getattr(self.model, 'model', None)
        if m is None:
            raise RuntimeError('Unexpected ultralytics model structure')

        first_conv = None
        name = None
        for name, module in m.named_modules():
            # ultralytics Conv wrapper often has attribute .conv which is nn.Conv2d
            if module.__class__.__name__ == 'Conv' and hasattr(module, 'conv'):
                conv = module.conv
                if conv.in_channels == 3:
                    first_conv = (module, conv)
                    break
            # otherwise look for raw nn.Conv2d
            if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                first_conv = (None, module)
                break

        if first_conv is None:
            raise RuntimeError('Could not find first conv layer to adapt')

        wrapper_module, original_conv = first_conv
        new_conv = torch.nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        torch.nn.init.kaiming_normal_(
            new_conv.weight, mode='fan_out', nonlinearity='relu')

        with torch.no_grad():
            if input_channels >= 3:
                new_conv.weight[:, :3, :, :].copy_(original_conv.weight)
            else:
                new_conv.weight.copy_(
                    original_conv.weight[:, :input_channels, :, :])
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias[:])

        if wrapper_module is not None:
            wrapper_module.conv = new_conv
        else:
            # We found a raw Conv2d instance; need to replace it on its parent.
            # Walk the model tree to set the attribute.
            parent = m
            parts = name.split('.')[:-1]
            for p in parts:
                parent = getattr(parent, p)
            setattr(parent, name.split('.')[-1], new_conv)

    def predict(self,
                imgs: Union[torch.Tensor, List[Union[str, torch.Tensor]]],
                conf: float = 0.25,
                iou: float = 0.45,
                imgsz: int = 640,
                augment: bool = False,
                device: Optional[Union[str, torch.device]] = None) -> List[dict]:
        """Run inference and return standardized detection outputs.

        Args:
            imgs: single image (Tensor or path) or list of images/paths
            conf: confidence threshold
            iou: NMS IoU threshold
            imgsz: inference size
            augment: whether to use augmentation during inference
            device: optional device override

        Returns:
            List of dicts (one per image) with keys: 'boxes', 'scores', 'labels', 'orig_size'
        """
        if device is not None:
            self.to(device)

        # Use Ultralytics' .predict API
        res = self.model.predict(
            source=imgs, conf=conf, iou=iou, imgsz=imgsz, augment=augment)

        outputs = []
        for r in res:  # r is a Results object
            # r.boxes.xyxy, r.boxes.conf, r.boxes.cls
            boxes = getattr(r.boxes, 'xyxy', None)
            scores = getattr(r.boxes, 'conf', None)
            labels = getattr(r.boxes, 'cls', None)

            # Convert to tensors if needed
            boxes = boxes.cpu() if boxes is not None else torch.empty((0, 4))
            scores = scores.cpu() if scores is not None else torch.empty((0,))
            labels = labels.cpu() if labels is not None else torch.empty((0,), dtype=torch.long)

            outputs.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels.long() if labels.numel() else labels,
                'orig_size': (int(r.orig_shape[0]), int(r.orig_shape[1])) if hasattr(r, 'orig_shape') else None,
                'result': r,  # raw result if caller needs it
            })

        return outputs

    # Keep a simple forward alias for integration convenience
    def forward(self, imgs, **kwargs):
        return self.predict(imgs, **kwargs)


def get_yolov8_detector(variant: str = 'small', pretrained: bool = True, **kwargs) -> YOLOv8Detector:
    """Factory returning a detection-ready YOLOv8Detector."""
    return YOLOv8Detector(variant=variant, pretrained=pretrained, **kwargs)


def yolo_nano(pretrained=True, **kwargs):
    return get_yolov8_detector('nano', pretrained=pretrained, **kwargs)


def yolo_small(pretrained=True, **kwargs):
    return get_yolov8_detector('small', pretrained=pretrained, **kwargs)


def yolo_medium(pretrained=True, **kwargs):
    return get_yolov8_detector('medium', pretrained=pretrained, **kwargs)


def yolo_large(pretrained=True, **kwargs):
    return get_yolov8_detector('large', pretrained=pretrained, **kwargs)


def yolo_xlarge(pretrained=True, **kwargs):
    return get_yolov8_detector('xlarge', pretrained=pretrained, **kwargs)
