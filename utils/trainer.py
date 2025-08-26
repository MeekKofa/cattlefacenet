"""
Clean and robust detection trainer.
Handles training loop, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
from .logger import Logger
from .timer import Timer
from .metrics import DetectionMetrics


class DetectionTrainer:
    """Clean trainer for object detection models."""

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Initialize components
        self.logger = Logger(config.get('log_dir', 'logs'), 'training')
        self.timer = Timer()
        self.metrics = DetectionMetrics(
            num_classes=config['num_classes'],
            iou_threshold=config.get('iou_threshold', 0.5)
        )

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.train_losses = []
        self.val_losses = []
        self.map_scores = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Trainer initialized for {config['num_classes']} classes")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")

    def _setup_optimizer(self):
        """Setup optimizer based on config."""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'step').lower()

        if scheduler_type == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = self.config.get('epochs', 100)
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            return None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.timer.start()

        # Create progress bar
        train_pbar = tqdm(self.train_loader, desc=f"Training Epoch",
                          leave=False, dynamic_ncols=True)

        for batch_idx, (images, targets) in enumerate(train_pbar):
            # Move to device - handle both tensor and list formats
            if isinstance(images, list):
                # If images is a list of tensors, stack them into a batch
                if len(images) > 0 and hasattr(images[0], 'to'):
                    images = torch.stack([img.to(self.device)
                                         for img in images])
                else:
                    # If already processed, just move to device
                    images = [img.to(self.device) for img in images]
            else:
                # If images is already a batched tensor
                images = images.to(self.device)

            targets = [{k: v.to(self.device) if hasattr(v, 'to') else v
                        for k, v in t.items()} for t in targets]

            # Forward pass
            self.optimizer.zero_grad()

            # Use model's built-in loss computation if available
            if hasattr(self.model, 'compute_loss'):
                # Model has built-in loss computation
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            else:
                # External loss computation needed
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar with current loss
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/(batch_idx+1):.4f}'
            })

        avg_loss = total_loss / num_batches
        epoch_time = self.timer.stop()

        # Close progress bar
        train_pbar.close()

        self.logger.info(
            f'Epoch {self.current_epoch} Training - '
            f'Loss: {avg_loss:.4f}, Time: {self.timer.format_time(epoch_time)}'
        )

        return avg_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # Move to device - handle both tensor and list formats
                if isinstance(images, list):
                    # If images is a list of tensors, stack them into a batch
                    if len(images) > 0 and hasattr(images[0], 'to'):
                        images = torch.stack(
                            [img.to(self.device) for img in images])
                    else:
                        # If already processed, just move to device
                        images = [img.to(self.device) for img in images]
                else:
                    # If images is already a batched tensor
                    images = images.to(self.device)

                targets = [{k: v.to(self.device) if hasattr(v, 'to') else v
                            for k, v in t.items()} for t in targets]

                # Forward pass - handle loss computation properly
                if hasattr(self.model, 'compute_loss'):
                    # During validation, we need both loss and predictions
                    # The model is in eval mode, so we need to force training mode temporarily for loss
                    was_training = self.model.training

                    # Get loss by temporarily switching to training mode
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    if isinstance(loss_dict, dict) and 'total_loss' in loss_dict:
                        loss = loss_dict['total_loss'].item()
                        total_loss += loss

                    # Get predictions in eval mode
                    self.model.eval()
                    # Returns detections for metrics
                    outputs = self.model(images)

                    # Debug: Check what outputs look like (first batch only)
                    if batch_idx == 0 and not hasattr(self, '_val_debug_printed'):
                        print(f"🔍 Validation Debug:")
                        print(f"  Outputs type: {type(outputs)}")
                        if isinstance(outputs, list) and len(outputs) > 0:
                            print(f"  First output type: {type(outputs[0])}")
                            if isinstance(outputs[0], dict):
                                print(
                                    f"  First output keys: {list(outputs[0].keys())}")
                                pred = outputs[0]
                                if 'boxes' in pred:
                                    print(
                                        f"  Predicted boxes shape: {pred['boxes'].shape}")
                                    print(
                                        f"  Num predicted boxes: {len(pred['boxes'])}")
                                    if len(pred['boxes']) > 0:
                                        print(
                                            f"  First 3 boxes: {pred['boxes'][:3]}")
                                if 'scores' in pred:
                                    print(
                                        f"  Predicted scores shape: {pred['scores'].shape}")
                                    print(
                                        f"  Num predicted scores: {len(pred['scores'])}")
                                    if len(pred['scores']) > 0:
                                        print(
                                            f"  Score range: {pred['scores'].min():.4f} - {pred['scores'].max():.4f}")
                                        print(
                                            f"  Scores above 0.1: {(pred['scores'] > 0.1).sum()}")
                                        print(
                                            f"  Scores above 0.5: {(pred['scores'] > 0.5).sum()}")
                                if 'labels' in pred:
                                    print(
                                        f"  Predicted labels shape: {pred['labels'].shape}")

                        print(f"  Targets sample: {len(targets)} targets")
                        if len(targets) > 0:
                            print(
                                f"  Target boxes: {targets[0]['boxes'].shape}")
                            print(
                                f"  Target labels: {targets[0]['labels'].shape}")
                        self._val_debug_printed = True

                    # Restore original training state
                    if was_training:
                        self.model.train()
                else:
                    # Standard forward pass
                    outputs = self.model(images)

                # Update metrics
                if not isinstance(outputs, dict):
                    self.metrics.update(outputs, targets)

        avg_loss = total_loss / len(self.val_loader) if total_loss > 0 else 0.0
        metrics = self.metrics.get_metrics()

        self.logger.info(
            f'Validation - Loss: {avg_loss:.4f}, '
            f'mAP@0.5: {metrics["mAP@0.5"]:.4f}'
        )

        return avg_loss, metrics

    def _compute_loss(self, outputs, targets):
        """Compute loss for models without built-in loss."""
        # Simple placeholder - should be implemented based on model type
        return torch.tensor(0.0, requires_grad=True)

    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'map_scores': self.map_scores,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.map_scores = checkpoint.get('map_scores', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f'Checkpoint loaded: {checkpoint_path}')

    def train(self, num_epochs):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            self.map_scores.append(metrics['mAP@0.5'])

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f'Learning rate: {current_lr:.6f}')

            # Save best model
            current_map = metrics['mAP@0.5']
            if current_map > self.best_map:
                self.best_map = current_map
                self.save_checkpoint('best_model.pth')
                self.logger.info(f'New best mAP: {self.best_map:.4f}')

            # Save periodic checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint()

            # Early stopping
            early_stop_patience = self.config.get('early_stop_patience', 0)
            if early_stop_patience > 0:
                if len(self.map_scores) > early_stop_patience:
                    recent_maps = self.map_scores[-early_stop_patience:]
                    if all(map_score <= self.best_map for map_score in recent_maps):
                        self.logger.info(
                            f'Early stopping after {epoch} epochs')
                        break

        self.logger.info(f"Training completed. Best mAP: {self.best_map:.4f}")

        # Save final checkpoint
        self.save_checkpoint('final_model.pth')

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'map_scores': self.map_scores,
            'best_map': self.best_map
        }

        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f'Training history saved: {history_path}')

        return history
