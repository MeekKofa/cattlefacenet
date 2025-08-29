"""YoloNet: simple combinator for YOLOv8 detector + ResNet backbone.

This module provides a lightweight `YoloNet` class that composes the
`YOLOv8Detector` (detection) and the `ResNetModel` (feature extractor)
from `model.backbone`. It implements two integration modes:

- 'ensemble' (default): run YOLO detections and optionally use ResNet to
  rescore or filter detections. This is fast and safe for experiments.
- 'fusion' (skeleton): placeholder for feature-level fusion. This requires
  additional design/training and is provided as a skeleton for research.

The implemented NMS merge is a simple utility used by the ensemble mode.
"""

from typing import List, Optional, Tuple, Dict, Any
import logging

import torch
import torch.nn.functional as F

# Import the backbones
from model.backbone.resnet import get_resnet, ResNetModel
from model.backbone.yolov8 import YOLOv8Detector, get_yolov8_detector

# YoloNet is an inference/experimental combinator only. Training-ready
# models must be created via ModelLoader (e.g. the 'yolonet_train' key).


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """Apply non-maximum suppression and return indices to keep.

    boxes: Tensor[N,4] in xyxy
    scores: Tensor[N]
    returns: indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    # torchvision.ops.nms would be ideal, but avoid adding dependency; implement simple NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
        if inds.numel() == 0:
            break
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.long)


class DetectorWrapper(torch.nn.Module):
    """nn.Module wrapper around the YOLOv8Detector to make it compatible with PyTorch APIs (DataParallel, .to, etc.)."""

    def __init__(self, detector):
        super(DetectorWrapper, self).__init__()
        self.detector = detector

    def forward(self, imgs, **kwargs):
        # Return the detector.predict output to keep behavior unchanged
        return self.detector.predict(imgs, **kwargs)


class YoloNet(torch.nn.Module):
    """Composition of YOLOv8 detector + ResNet feature extractor.

    Args:
        yolo_variant: str, YOLO variant passed to YOLOv8Detector factory
        resnet_depth: int, ResNet depth (18,34,50,...)
        mode: 'ensemble' or 'fusion'
    """

    def __init__(self, yolo_variant: str = 'small', resnet_depth: int = 50, mode: str = 'ensemble', device: Optional[str] = None, num_classes: Optional[int] = None, dataset_name: Optional[str] = None):
        super(YoloNet, self).__init__()

        # Basic configuration
        self.device = device
        self.mode = mode
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.resnet_depth = resnet_depth

        # Placeholder for a training-ready detection model (created lazily)
        self.training_model = None

        # Detector (wrapped as nn.Module)
        try:
            detector = get_yolov8_detector(yolo_variant, pretrained=True)
            self.detector = DetectorWrapper(detector)
            # keep reference to raw detector for .to() forwarding
            self._raw_detector = detector
        except Exception as e:
            logging.warning('Failed to init YOLOv8 detector: %s', e)
            self.detector = None
            self._raw_detector = None

        # ResNet feature extractor (we want features -> use forward_without_fc)
        try:
            self.resnet = get_resnet(
                resnet_depth, pretrained=True, num_classes=1000)
        except Exception as e:
            logging.warning('Failed to init ResNet: %s', e)
            self.resnet = None

        # Move to device if requested
        if self.device is not None:
            try:
                self.to(self.device)
            except Exception:
                pass

    def to(self, device):
        """Move all subcomponents to device and return self."""
        dev = torch.device(device) if not isinstance(
            device, torch.device) else device
        if self.detector is not None and self._raw_detector is not None:
            try:
                # YOLOv8Detector implements .to
                self._raw_detector.to(dev)
            except Exception:
                pass
        if self.resnet is not None:
            self.resnet.to(dev)
        super(YoloNet, self).to(dev)
        self.device = dev
        return self

    def half(self):
        """Convert internal modules to half precision where supported."""
        if self.detector is not None and self._raw_detector is not None:
            try:
                if hasattr(self._raw_detector, 'model'):
                    self._raw_detector.model.half()
            except Exception:
                pass
        if self.resnet is not None:
            self.resnet.half()
        return self

    def train(self, mode: bool = True):
        if self.detector is not None:
            try:
                # detector wrapper has no trainable params usually
                self.detector.train(mode)
            except Exception:
                pass
        if self.resnet is not None:
            self.resnet.train(mode)
        return super(YoloNet, self).train(mode)

    def eval(self):
        if self.detector is not None:
            try:
                self.detector.eval()
            except Exception:
                pass
        if self.resnet is not None:
            self.resnet.eval()
        return super(YoloNet, self).eval()

    def predict(self, imgs, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640) -> List[Dict[str, Any]]:
        """Run the composed model and return merged detection outputs.

        Outputs are a list of dicts like the underlying detector with potential
        added keys like 'resnet_feats' when mode=='fusion' is used.
        """
        if self.detector is None or self._raw_detector is None:
            raise RuntimeError('YOLO detector is not initialized')

        # Use the raw detector predict to preserve API
        yolo_outs = self._raw_detector.predict(
            imgs, conf=conf, iou=iou, imgsz=imgsz)

        results = []
        for idx, out in enumerate(yolo_outs):
            boxes = out['boxes']
            scores = out['scores']
            labels = out['labels']

            # Simple postprocessing: apply NMS again to ensure cleanliness
            keep = nms_boxes(boxes, scores, iou_threshold=iou)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            entry = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'orig_size': out.get('orig_size'),
                'raw': out.get('result')
            }

            # Fusion mode: extract ResNet features for each detection region (skeleton)
            if self.mode == 'fusion' and self.resnet is not None:
                try:
                    # Crop and resize each box region, run through resnet forward_without_fc
                    im_tensor = None
                    # If the detector returned the raw result with image tensor, we could use it.
                    # For now we leave a placeholder to avoid heavy IO here.
                    resnet_feats = []
                    for b in boxes:
                        # placeholder zeros per box
                        resnet_feats.append(torch.zeros(
                            1, 512 * self.resnet.block_expansion, 1, 1))
                    entry['resnet_feats'] = torch.cat(
                        resnet_feats, dim=0) if resnet_feats else torch.empty((0,))
                except Exception as e:
                    logging.warning('Fusion extraction failed: %s', e)

            results.append(entry)

        return results

    def forward(self, imgs, targets=None, **kwargs):
        """Forward compatibility for trainer: return detections for given images.

        The trainer expects model(images) to return detection outputs when the
        model does not implement `compute_loss`. We provide that here by
        delegating to `predict`.
        """
        # YoloNet is inference-only. If targets are provided the caller is
        # likely attempting training; in that case the correct behavior is
        # to load a training-ready model from ModelLoader (for example the
        # 'yolonet_train' key) and train that model. We raise a clear error
        # to avoid silently creating models here and to keep ModelLoader as
        # the single source of truth for model construction.
        if targets is not None:
            raise RuntimeError(
                "YoloNet is inference-only and does not implement training/compute_loss. "
                "Use ModelLoader to create a training-ready model (e.g. 'yolonet_train') "
                "and pass that to the trainer.")

        return self.predict(imgs, **kwargs)

    # Note: compute_loss intentionally not implemented here to keep the
    # model loader as the single source of truth for creating training-ready
    # models. Use ModelLoader.get_model('yolonet_train', ...) to obtain a
    # differentiable detection model suitable for training.


def build_yolonet_default(device: Optional[str] = None) -> YoloNet:
    """Utility to create a default Yolonet instance (small+x50, ensemble)."""
    return YoloNet(yolo_variant='small', resnet_depth=50, mode='ensemble', device=device)


def get_yolonet(input_channels: int = 3,
                num_classes: int = 400,
                pretrained: bool = False,
                resnet_depth: int = 50,
                yolo_variant: str = 'small',
                mode: str = 'ensemble',
                device: Optional[str] = None,
                **kwargs) -> YoloNet:
    """Factory for creating a YoloNet instance compatible with ModelLoader.

    Parameters are intentionally flexible to match how ModelLoader passes kwargs.
    """
    # Note: `pretrained` is accepted for API compatibility but is currently only
    # used by the underlying YOLO and ResNet factories if applicable.
    net = YoloNet(yolo_variant=yolo_variant,
                  resnet_depth=resnet_depth, mode=mode, device=device)
    return net
