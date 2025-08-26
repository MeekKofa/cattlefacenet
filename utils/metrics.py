"""
Detection metrics for object detection evaluation.
Clean implementation with mAP calculation.
"""

import torch
import numpy as np
from collections import defaultdict


class DetectionMetrics:
    """Clean implementation of detection metrics including mAP."""

    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []

    def update(self, predictions, targets):
        """Update metrics with batch predictions and targets.

        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute_map(self):
        """Compute mean Average Precision."""
        if not self.predictions or not self.targets:
            return 0.0

        # Collect all predictions and ground truths
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        all_image_ids = []

        for i, (pred, gt) in enumerate(zip(self.predictions, self.targets)):
            if len(pred['boxes']) > 0:
                all_pred_boxes.append(pred['boxes'].cpu())
                all_pred_scores.append(pred['scores'].cpu())
                all_pred_labels.append(pred['labels'].cpu())
                all_image_ids.extend([i] * len(pred['boxes']))

            if len(gt['boxes']) > 0:
                all_gt_boxes.append(gt['boxes'].cpu())
                all_gt_labels.append(gt['labels'].cpu())

        if not all_pred_boxes or not all_gt_boxes:
            return 0.0

        # Concatenate all predictions and ground truths
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)
        pred_labels = torch.cat(all_pred_labels, dim=0)

        # Calculate AP for each class
        aps = []
        for class_id in range(self.num_classes):
            class_pred_mask = pred_labels == class_id
            if not class_pred_mask.any():
                continue

            class_pred_boxes = pred_boxes[class_pred_mask]
            class_pred_scores = pred_scores[class_pred_mask]

            # Get ground truth boxes for this class
            class_gt_boxes = []
            for i, gt in enumerate(self.targets):
                if len(gt['boxes']) > 0:
                    gt_mask = gt['labels'] == class_id
                    if gt_mask.any():
                        class_gt_boxes.append(gt['boxes'][gt_mask].cpu())

            if not class_gt_boxes:
                continue

            class_gt_boxes = torch.cat(class_gt_boxes, dim=0)

            # Calculate AP for this class
            ap = self._calculate_ap(
                class_pred_boxes, class_pred_scores, class_gt_boxes)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def _calculate_ap(self, pred_boxes, pred_scores, gt_boxes):
        """Calculate Average Precision for a single class."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0

        # Sort predictions by confidence score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))

        # Track which ground truth boxes have been matched
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i, pred_box in enumerate(pred_boxes):
            # Calculate IoU with all ground truth boxes
            ious = self._calculate_iou(pred_box.unsqueeze(0), gt_boxes)
            max_iou, max_idx = torch.max(ious, dim=1)
            max_iou = max_iou.item()
            max_idx = max_idx.item()

            if max_iou >= self.iou_threshold and not gt_matched[max_idx]:
                tp[i] = 1
                gt_matched[max_idx] = True
            else:
                fp[i] = 1

        # Calculate precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / len(gt_boxes)

        # Calculate AP using 11-point interpolation
        ap = self._calculate_ap_11_point(precision, recall)
        return ap.item()

    def _calculate_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes."""
        # boxes format: [x1, y1, x2, y2]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Calculate intersection
        x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
        y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
        x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
        y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)

        intersection = torch.clamp(x2 - x1, min=0) * \
            torch.clamp(y2 - y1, min=0)
        union = area1[:, None] + area2[None, :] - intersection

        return intersection / (union + 1e-8)

    def _calculate_ap_11_point(self, precision, recall):
        """Calculate AP using 11-point interpolation."""
        recall_thresholds = torch.linspace(0, 1, 11)
        ap = torch.zeros(11)

        for i, recall_thresh in enumerate(recall_thresholds):
            precisions_above_thresh = precision[recall >= recall_thresh]
            if len(precisions_above_thresh) > 0:
                ap[i] = torch.max(precisions_above_thresh)

        return torch.mean(ap)

    def get_metrics(self):
        """Get all computed metrics."""
        map_50 = self.compute_map()
        return {
            'mAP@0.5': map_50,
            'num_predictions': len(self.predictions),
            'num_targets': len(self.targets)
        }
