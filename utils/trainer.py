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

        for batch_idx, (images, targets) in enumerate(self.train_loader):
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

            # Log progress less frequently and only to one logger
            if batch_idx % self.config.get('log_interval', 100) == 0:
                progress = 100.0 * batch_idx / num_batches
                # Only log to main logger, not both
                print(f'Epoch {self.current_epoch}: [{batch_idx}/{num_batches} '
                      f'({progress:.1f}%)] Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        epoch_time = self.timer.stop()

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
            for images, targets in self.val_loader:
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

                # Forward pass
                if self.model.training:
                    self.model.eval()

                outputs = self.model(images)

                # Compute validation loss if possible
                if hasattr(self.model, 'compute_loss'):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    total_loss += loss.item()

                # Update metrics
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
