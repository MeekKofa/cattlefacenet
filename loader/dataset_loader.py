"""
Centralized Dataset Loader - Single Source of Truth for Dataset Management
This is the ONLY file that should know about dataset paths and structure.
All other files should use this module to access datasets.

This module also includes multiprocessing handling and training optimizations.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, Tuple, Optional
import os
from PIL import Image
from loader.object_detection_dataset import ObjectDetectionDataset, collate_fn_object_detection, vgg_yolo_collate_fn
from class_mapping import CLASS_MAPPING, get_num_classes

# Training and system configuration constants
TRAINING_DEFAULTS = {
    'detection_conf_threshold': 0.25,  # Lower threshold for better detection
    'nms_iou_threshold': 0.45,
    'max_detections_per_image': 300,
    'gradient_clip_value': 1.0,
    'warmup_epochs': 5,
    'loss_weights': {
        'box_loss_weight': 5.0,    # Emphasize box regression
        'obj_loss_weight': 1.0,    # Objectness
        'cls_loss_weight': 2.0     # Classification
    }
}

# SINGLE SOURCE OF TRUTH - Dataset Configuration Dictionary
DATASET_REGISTRY = {
    'cattleface': {
        'base_path': 'processed_data/cattleface',
        'modality': 'object_detection',
        'structure': {
            'images_subdir': 'images',
            'labels_subdir': 'annotations'  # Your actual structure
        },
        # Based on class IDs seen (up to ~310+), use larger number for safety
        'num_classes': 400,
        'splits': ['train', 'val', 'test'],
        'image_extensions': ['.jpg', '.jpeg', '.png'],
        'target_size': (448, 448),
        'description': 'Cattle face detection dataset with processed annotations'
    },
    'cattlebody': {
        'base_path': 'processed_data/cattlebody',
        'modality': 'object_detection',
        'structure': {
            'images_subdir': 'images',
            'labels_subdir': 'annotations'
        },
        'num_classes': 400,  # Same as cattleface for consistency
        'splits': ['train', 'val', 'test'],
        'image_extensions': ['.jpg', '.jpeg', '.png'],
        'target_size': (448, 448),
        'description': 'Cattle body detection dataset'
    }
}


def get_training_defaults():
    """Get default training configuration."""
    return TRAINING_DEFAULTS.copy()


def get_safe_dataloader_kwargs(num_workers: int = 4, pin_memory: bool = True):
    """Get DataLoader kwargs with multiprocessing safety."""
    return {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': False,  # Disable for stability
        'drop_last': True  # For stable batch sizes
    }


def apply_multiprocessing_fixes():
    """Apply system-level fixes for multiprocessing issues."""
    # Set environment variables for stability
    env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:32',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1'
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # Set multiprocessing method
    try:
        import torch.multiprocessing as mp
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logging.warning(f"Could not set multiprocessing method: {e}")

    # Set PyTorch sharing strategy if available
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except AttributeError:
        pass

    logging.info("✅ Multiprocessing fixes applied")


def object_detection_collate(batch):
    """Collate function for object detection datasets."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.zeros(0), torch.zeros(0)

    images, targets = zip(*batch)
    images = torch.stack(images)
    formatted_targets = []
    for target in targets:
        formatted = {
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id']
        }
        formatted_targets.append(formatted)
    return images, formatted_targets


class CentralizedDatasetLoader:
    """
    Centralized dataset loader - the ONLY class that should know about dataset paths.
    This ensures all dataset access goes through one consistent interface.
    """

    def __init__(self):
        self.registry = DATASET_REGISTRY

    def list_available_datasets(self) -> Dict[str, str]:
        """Get list of all available datasets with their descriptions"""
        return {name: config['description'] for name, config in self.registry.items()}

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get complete configuration for a dataset"""
        return self.registry.get(dataset_name.lower())

    def is_dataset_available(self, dataset_name: str) -> bool:
        """Check if a dataset is registered and available"""
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            return False

        base_path = Path(dataset_info['base_path'])
        return base_path.exists()

    def validate_dataset_structure(self, dataset_name: str) -> Tuple[bool, str]:
        """Validate that a dataset has the required directory structure"""
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            return False, f"Dataset '{dataset_name}' not registered"

        base_path = Path(dataset_info['base_path'])
        if not base_path.exists():
            return False, f"Dataset base path not found: {base_path}"

        structure = dataset_info['structure']
        splits = dataset_info['splits']

        for split in splits:
            split_path = base_path / split
            images_dir = split_path / structure['images_subdir']
            labels_dir = split_path / structure['labels_subdir']

            if not images_dir.exists():
                return False, f"Images directory not found: {images_dir}"

            if not labels_dir.exists():
                return False, f"Labels directory not found: {labels_dir}"

            # Check for actual image files (only for training split)
            if split == 'train':
                extensions = dataset_info['image_extensions']
                has_images = any(
                    f.suffix.lower() in extensions
                    for f in images_dir.iterdir() if f.is_file()
                )
                if not has_images:
                    return False, f"No valid image files found in: {images_dir}"

        return True, "Dataset structure is valid"

    def get_dataset_paths(self, dataset_name: str) -> Dict[str, Dict[str, Path]]:
        """Get structured paths for all splits of a dataset"""
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not registered")

        base_path = Path(dataset_info['base_path'])
        structure = dataset_info['structure']
        splits = dataset_info['splits']

        paths = {}
        for split in splits:
            split_path = base_path / split
            paths[split] = {
                'images': split_path / structure['images_subdir'],
                'labels': split_path / structure['labels_subdir']
            }

        return paths

    def load_dataset(self, dataset_name: str, batch_size: Dict[str, int],
                     num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load a complete dataset with train/val/test loaders.
        This is the MAIN interface that other files should use.
        """
        logging.info(f"Loading dataset: {dataset_name}")

        # Validate dataset
        is_valid, message = self.validate_dataset_structure(dataset_name)
        if not is_valid:
            raise FileNotFoundError(f"Dataset validation failed: {message}")

        # Get dataset info and paths
        dataset_info = self.get_dataset_info(dataset_name)
        dataset_paths = self.get_dataset_paths(dataset_name)
        target_size = dataset_info['target_size']

        # Create datasets
        train_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['train']['images']),
            annotation_dir=str(dataset_paths['train']['labels']),
            transform=self._get_transforms(target_size, is_train=True)
        )
        val_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['val']['images']),
            annotation_dir=str(dataset_paths['val']['labels']),
            transform=self._get_transforms(target_size, is_train=False)
        )
        test_dataset = ObjectDetectionDataset(
            image_dir=str(dataset_paths['test']['images']),
            annotation_dir=str(dataset_paths['test']['labels']),
            transform=self._get_transforms(target_size, is_train=False)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size['val'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=object_detection_collate
        )

        # Log dataset info
        logging.info(f"✅ Dataset '{dataset_name}' loaded successfully:")
        logging.info(f"   📁 Base path: {dataset_info['base_path']}")
        logging.info(f"   🔢 Classes: {dataset_info['num_classes']}")
        logging.info(f"   📊 Train: {len(train_dataset)} images")
        logging.info(f"   📊 Val: {len(val_dataset)} images")
        logging.info(f"   📊 Test: {len(test_dataset)} images")
        logging.info(
            f"   🏷️  Detected classes: {len(sorted(train_dataset.classes))} classes ({min(train_dataset.classes)}-{max(train_dataset.classes)})")

        return train_loader, val_loader, test_loader

    def _get_transforms(self, target_size: Tuple[int, int], is_train: bool = True):
        """Get image transforms for the dataset"""
        # Force use of torchvision transforms for compatibility
        # TODO: Implement proper Albumentations bbox transform integration later
        logging.warning("Using torchvision transforms for compatibility")
        from torchvision import transforms

        if is_train:
            return transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])


# Global dataset loader instance - single source of truth
_dataset_loader = None


def get_dataset_loader() -> CentralizedDatasetLoader:
    """Get the global dataset loader instance"""
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = CentralizedDatasetLoader()
    return _dataset_loader

# Public API functions - these are what other files should use


def load_dataset(dataset_name: str, batch_size: Dict[str, int],
                 num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Main function to load any dataset - use this from other files"""
    return get_dataset_loader().load_dataset(dataset_name, batch_size, num_workers, pin_memory)


def list_datasets() -> Dict[str, str]:
    """List all available datasets"""
    return get_dataset_loader().list_available_datasets()


def is_dataset_available(dataset_name: str) -> bool:
    """Check if a dataset is available"""
    return get_dataset_loader().is_dataset_available(dataset_name)


def get_dataset_info(dataset_name: str) -> Optional[Dict]:
    """Get dataset configuration info"""
    return get_dataset_loader().get_dataset_info(dataset_name)


def load_dataset_vgg_yolo(dataset_name: str, batch_size: Dict[str, int],
                          num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load dataset with VGG YOLO compatible collate function"""
    # Get the datasets using the regular loader
    loader = get_dataset_loader()

    # Validate dataset
    if not loader.is_dataset_available(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in registry")

    # Get dataset info and paths (use the same method as regular loader)
    dataset_info = loader.get_dataset_info(dataset_name)
    dataset_paths = loader.get_dataset_paths(dataset_name)
    target_size = dataset_info['target_size']

    # Create datasets using the same approach as regular loader
    train_dataset = ObjectDetectionDataset(
        image_dir=str(dataset_paths['train']['images']),
        annotation_dir=str(dataset_paths['train']['labels']),
        transform=loader._get_transforms(target_size, is_train=True)
    )

    val_dataset = ObjectDetectionDataset(
        image_dir=str(dataset_paths['val']['images']),
        annotation_dir=str(dataset_paths['val']['labels']),
        transform=loader._get_transforms(target_size, is_train=False)
    )

    test_dataset = ObjectDetectionDataset(
        image_dir=str(dataset_paths['test']['images']),
        annotation_dir=str(dataset_paths['test']['labels']),
        transform=loader._get_transforms(target_size, is_train=False)
    )

    # Create data loaders with VGG YOLO collate function
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=vgg_yolo_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size['val'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=vgg_yolo_collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=vgg_yolo_collate_fn
        )

        return train_loader, val_loader, test_loader

    except RuntimeError as e:
        if "shared memory" in str(e).lower() or "multiprocessing" in str(e).lower():
            logging.warning(f"Multiprocessing DataLoader failed: {e}")
            logging.warning("Retrying with num_workers=0 and pin_memory=False")

            # Retry with single-threaded setup
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size['train'],
                shuffle=True,
                num_workers=0,  # Disable multiprocessing
                pin_memory=False,  # Disable pin_memory
                collate_fn=vgg_yolo_collate_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size['val'],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=vgg_yolo_collate_fn
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size['test'],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=vgg_yolo_collate_fn
            )

            return train_loader, val_loader, test_loader
        else:
            # Re-raise if it's not a multiprocessing issue
            raise e

    return train_loader, val_loader, test_loader


def validate_dataset(dataset_name: str) -> Tuple[bool, str]:
    """Validate dataset structure"""
    return get_dataset_loader().validate_dataset_structure(dataset_name)


# Legacy compatibility - keeping old class name for backward compatibility
class DatasetLoader(CentralizedDatasetLoader):
    """Backward compatibility alias"""

    def load_data(self, *args, **kwargs):
        """Legacy method - redirects to new load_dataset"""
        logging.warning(
            "DatasetLoader.load_data() is deprecated, use load_dataset() function instead")
        return self.load_dataset(*args, **kwargs)
