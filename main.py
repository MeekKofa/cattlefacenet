"""
Clean main entry point for cattle face detection training and evaluation.
Provides a streamlined workflow with robust error handling.
"""

import os
import torch
import torchvision
import logging
import argparse
from pathlib import Path
import json

from loader.dataset_loader import DatasetLoader
from model.model_loader import ModelLoader
from utils.trainer import DetectionTrainer
from utils.logger import Logger
from argument_parser import parse_args


def setup_device(gpu_ids=None):
    """Setup computing device."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')

    if gpu_ids:
        device_id = gpu_ids[0] if isinstance(gpu_ids, list) else gpu_ids
        device = torch.device(f'cuda:{device_id}')

        # Get the actual physical GPU ID from environment
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        physical_gpu_id = visible_devices.split(
            ',')[device_id] if ',' in visible_devices else visible_devices

        print(f"Using CUDA device: {device} (Physical GPU {physical_gpu_id})")
        print(
            f"✅ Correctly using Physical GPU {physical_gpu_id} (mapped to {device})")

        # Print device info and memory status
        props = torch.cuda.get_device_properties(device)
        print(f"Device: {props.name}")
        print(f"Memory: {props.total_memory / 1024**3:.2f} GB")

        # Clear cache first
        torch.cuda.empty_cache()

        # Check memory usage after clearing cache
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            memory_free = (props.total_memory / 1024**3) - memory_reserved
            print(f"Memory allocated: {memory_allocated:.2f} GB")
            print(f"Memory reserved: {memory_reserved:.2f} GB")
            print(f"Memory free: {memory_free:.2f} GB")

            # Warn if low memory
            if memory_free < 2.0:  # Less than 2GB free
                print(
                    "⚠️  WARNING: Low GPU memory available. Consider reducing batch size.")
    else:
        device = torch.device('cuda:0')
        print(f"Using default CUDA device: {device}")

    return device


def auto_adjust_batch_size(initial_batch_size, device):
    """Auto-adjust batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return initial_batch_size

    try:
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        memory_free = total_memory - memory_reserved

        # Conservative estimate for ResNet50-YOLO: each sample needs ~200MB
        # This accounts for model parameters, activations, and gradients
        estimated_memory_per_sample = 0.2  # GB
        # Use only 60% of free memory to be conservative
        max_batch_size = int(memory_free * 0.6 / estimated_memory_per_sample)

        # Cap maximum batch size for very large memory
        max_batch_size = min(max_batch_size, 16)  # Cap at 16 for ResNet50-YOLO

        if max_batch_size < initial_batch_size:
            print(
                f"⚠️  Reducing batch size from {initial_batch_size} to {max_batch_size} due to memory constraints")
            return max(max_batch_size, 1)  # At least batch size of 1

        return initial_batch_size

    except Exception as e:
        print(f"Warning: Could not auto-adjust batch size: {e}")
        return initial_batch_size


def create_training_config(args):
    """Create training configuration from arguments."""

    # Handle depth configuration
    if isinstance(args.depth, str):
        try:
            depth_config = json.loads(args.depth)
        except json.JSONDecodeError:
            depth_config = {}
    else:
        depth_config = args.depth

    # Handle data argument (could be string or list)
    dataset_name = args.data[0] if isinstance(args.data, list) else args.data

    config = {
        # Model and data
        'num_classes': 400,  # Cattle face classes
        'arch': args.arch,
        'dataset': dataset_name,
        'batch_size': args.train_batch,

        # Training parameters
        'epochs': args.epochs,
        # Ensure minimum LR for object detection
        'learning_rate': max(args.lr, 0.001),
        'optimizer': args.optimizer,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # Scheduler
        'scheduler': 'step',
        'step_size': 30,
        'gamma': 0.1,

        # Detection specific
        'iou_threshold': 0.5,
        'confidence_threshold': 0.25,
        'nms_threshold': 0.45,

        # Training settings
        'gradient_clip': 1.0,
        'dropout_rate': args.drop,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,

        # Logging and saving
        'log_interval': 20,
        'save_interval': 10,
        'early_stop_patience': 15,
        'log_dir': f'logs/{args.task_name}',
        'checkpoint_dir': f'checkpoints/{args.task_name}',

        # Model specific
        'depth': depth_config,
        'gpu_ids': args.gpu_ids
    }

    return config


def setup_data(config, logger):
    """Setup data loaders."""
    logger.info(f"Setting up dataset: {config['dataset']}")

    # Initialize dataset loader
    dataset_loader = DatasetLoader()

    # Prepare batch size configuration
    batch_sizes = {
        'train': config['batch_size'],
        'val': config['batch_size'],
        'test': config['batch_size']
    }

    # Load dataset
    train_loader, val_loader, test_loader = dataset_loader.load_dataset(
        dataset_name=config['dataset'],
        batch_size=batch_sizes,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    return train_loader, val_loader


def setup_model(config, device, logger):
    """Setup model."""
    logger.info(f"Setting up model: {config['arch']}")

    # Get the architecture name (it's a list, so take the first element)
    arch_name = config['arch'][0] if isinstance(
        config['arch'], list) else config['arch']

    # Initialize model loader
    model_loader = ModelLoader(device=device, arch=arch_name)

    # Create model
    models_and_names = model_loader.get_model(
        model_name=arch_name,
        depth=config.get('depth', {}),
        input_channels=3,
        num_classes=config['num_classes'],
        task_name=None,  # No pre-trained loading for training
        dataset_name=None
    )

    if not models_and_names:
        raise ValueError("No models returned from model loader")

    model, model_name = models_and_names[0]

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def train_model(args):
    """Main training function."""
    # Setup
    device = setup_device(args.gpu_ids_list)

    # Auto-adjust batch size based on GPU memory
    original_batch_size = args.train_batch
    adjusted_batch_size = auto_adjust_batch_size(original_batch_size, device)
    args.train_batch = adjusted_batch_size

    config = create_training_config(args)

    # Create directories
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = Path(config['checkpoint_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize logger
    logger = Logger(config['log_dir'], 'training')
    logger.info("="*60)
    logger.info("CATTLE FACE DETECTION - TRAINING START")
    logger.info("="*60)
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Architecture: {config['arch']}")
    logger.info(f"Device: {device}")

    try:
        # Setup data
        train_loader, val_loader = setup_data(config, logger)

        # Setup model
        model = setup_model(config, device, logger)

        # Initialize trainer
        trainer = DetectionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )

        # Start training
        logger.info("Starting training...")
        history = trainer.train(config['epochs'])

        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Best mAP@0.5: {history['best_map']:.4f}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error("Check the logs for more details")
        raise


def evaluate_model(args):
    """Evaluation function."""
    # Setup
    device = setup_device(args.gpu_ids_list)
    config = create_training_config(args)

    logger = Logger('logs/evaluation', 'evaluation')
    logger.info("Starting model evaluation...")

    try:
        # Setup data
        _, val_loader = setup_data(config, logger)

        # Setup model
        model = setup_model(config, device, logger)

        # Load checkpoint
        checkpoint_path = f"checkpoints/{args.task_name}/best_model.pth"
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        # Evaluate
        trainer = DetectionTrainer(model, None, val_loader, config, device)
        _, metrics = trainer.validate()

        logger.info("="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Log basic system information (simplified)
    print(f"🔧 System: PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 GPUs: {torch.cuda.device_count()} devices")
    
    # Configure logging to reduce verbosity
    logging.basicConfig(level=logging.WARNING)

    print("="*80)
    print("CATTLE FACE DETECTION SYSTEM")
    print("="*80)
    print(f"Task: {args.task_name}")
    print(f"Dataset: {args.data}")
    print(f"Architecture: {args.arch}")

    # Determine if this is training or evaluation based on model_path
    is_evaluation = hasattr(args, 'model_path') and args.model_path is not None
    print(f"Mode: {'Evaluation' if is_evaluation else 'Training'}")
    print("="*80)

    try:
        if is_evaluation:
            evaluate_model(args)
        else:
            train_model(args)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Check the logs for more details")
        raise


if __name__ == "__main__":
    main()
