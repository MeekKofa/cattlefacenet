# model_loader.py

import os
import torch
import logging

from model.backbone.resnet import get_resnet
from model.backbone.densenet import get_densenet
from model.backbone.vgg import get_vgg
from model.yolo.vggnet import get_vggnet
from model.yolo.yolov8vgg import get_yolov8vgg
from model.yolo.yolov8resnet import get_yolo8resnet
from model.yolo.yolonet.yolonet import get_yolonet
from model.attention.MSARNet import MSARNet
from model.meddef.meddef1 import get_meddef1
from utils.memory_efficient_model import MemoryEfficientModel
# Optional import to infer num_classes from registered datasets when not provided
try:
    from loader.dataset_loader import get_dataset_loader
except Exception:
    get_dataset_loader = None


class ModelLoader:
    def __init__(self, device, arch, pretrained=True, fp16=False):
        self.device = device
        self.arch = arch
        self.pretrained = pretrained
        self.fp16 = fp16  # New flag for FP16 conversion

        # Define model architectures and their depths
        self.models_dict = {
            'resnet': {'func': get_resnet, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes'], 'type': 'classification'},
            'densenet': {'func': get_densenet, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes'], 'type': 'classification'},
            'vgg': {'func': get_vgg, 'params': ['depth', 'pretrained', 'input_channels', 'num_classes'], 'type': 'classification'},
            'vggnet': {'func': get_vggnet, 'params': ['depth', 'pretrained=False', 'input_channels', 'num_classes'], 'type': 'classification'},
            # Public 'yolonet' should map to the research combinator (inference/experiments)
            'yolonet': {'func': get_yolonet, 'params': ['input_channels', 'num_classes', 'pretrained', 'resnet_depth', 'yolo_variant', 'mode', 'device'], 'type': 'object_detection'},
            # Separate key for training-ready implementation (provides compute_loss)
            'yolonet_train': {'func': get_yolo8resnet, 'params': ['input_channels', 'num_classes', 'pretrained'], 'type': 'object_detection'},
            'meddef1': {'func': get_meddef1, 'params': ['depth', 'input_channels', 'num_classes', 'robust_method'], 'type': 'classification'},
        }
        logging.info("ModelLoader initialized with models: " +
                     ", ".join(self.models_dict.keys()))

    def is_classification_model(self, model_name):
        """Check if a model is a classification model."""
        if model_name in self.models_dict:
            return self.models_dict[model_name].get('type') == 'classification'
        return False

    def _format_model_name(self, model_name, depth):
        """Format model name with depth in a filename-friendly way"""
        if isinstance(depth, dict):
            # Extract the depth values for this specific model
            model_depths = depth.get(model_name, [])
            return f"{model_name}_{', '.join(map(str, model_depths))}"
        return f"{model_name}_{depth}"

    def get_latest_checkpoint(self, model_name_with_depth, dataset_name, load_task, adversarial=False):
        """Finds the most recent checkpoint for the given model and dataset.

        Args:
            model_name_with_depth: Formatted model name with depth
            dataset_name: Name of the dataset
            load_task: Task name (e.g., 'normal_training', 'attack', 'defense')
            adversarial: Whether to look for adversarial checkpoints
        """
        # Construct path based on whether it's adversarial or normal training
        if adversarial:
            checkpoint_dir = f"out/{load_task}/{dataset_name}/{model_name_with_depth}/adv/save_model"
            logging.info(
                f"Looking for adversarial checkpoint in: {checkpoint_dir}")
        else:
            checkpoint_dir = f"out/{load_task}/{dataset_name}/{model_name_with_depth}/save_model"
            logging.info(f"Looking for normal checkpoint in: {checkpoint_dir}")

        if not os.path.exists(checkpoint_dir):
            logging.info(
                f"ModelLoader: No checkpoint directory found for {model_name_with_depth} in {checkpoint_dir}")

            # If adversarial checkpoint not found, try fallback to non-adversarial
            if adversarial:
                fallback_dir = f"out/{load_task}/{dataset_name}/{model_name_with_depth}/save_model"
                logging.info(
                    f"Trying fallback to normal checkpoint directory: {fallback_dir}")
                if os.path.exists(fallback_dir):
                    checkpoint_dir = fallback_dir
                    logging.info(f"Using fallback normal checkpoint directory")
                else:
                    return None
            else:
                return None

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(
            f"best_{model_name_with_depth}_{dataset_name}")]
        if not checkpoints:
            logging.info(
                f"ModelLoader: No checkpoints found for {model_name_with_depth} in {checkpoint_dir}")
            return None

        # Sort checkpoints by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(
            os.path.join(checkpoint_dir, x)), reverse=True)
        latest_checkpoint = checkpoints[0]
        logging.info(f"Selected checkpoint: {latest_checkpoint}")
        return os.path.join(checkpoint_dir, latest_checkpoint)

    def get_model(self, model_name=None, depth=None, input_channels=3, num_classes=None, task_name=None,
                  dataset_name=None, adversarial=False):
        """Retrieves model(s) based on specified architecture, depth, and configurations."""
        model_name = model_name or self.arch

        if model_name not in self.models_dict:
            raise ValueError(f"Model {model_name} not recognized.")

        # If num_classes is not provided, try to infer it from the dataset registry
        if num_classes is None:
            inferred = None
            if dataset_name and get_dataset_loader is not None:
                try:
                    ds_loader = get_dataset_loader()
                    info = ds_loader.get_dataset_info(dataset_name)
                    if info and 'num_classes' in info and info['num_classes']:
                        inferred = info['num_classes']
                except Exception:
                    inferred = None

            if inferred is not None:
                num_classes = inferred
                logging.info(
                    f"ModelLoader: Inferred num_classes={num_classes} from dataset registry for dataset '{dataset_name}'")
            else:
                raise ValueError("num_classes must be specified")

        model_entry = self.models_dict[model_name]
        model_func = model_entry['func']
        model_params = model_entry['params']

        # Handle multiple depths
        # Allow passing a depth dict keyed by a related/base model name.
        # Example: depth={'yolonet':[50]} but model_name='yolonet_train'
        if isinstance(depth, dict):
            model_depths = depth.get(model_name, [])
            if not model_depths:
                # Try to find a base key by stripping common suffixes like '_train'
                base_name = model_name
                if model_name.endswith('_train'):
                    base_name = model_name[: -len('_train')]
                # If base present in provided depth dict, use it
                if base_name in depth:
                    model_depths = depth.get(base_name, [])

            if not model_depths:
                raise ValueError(
                    f"No depths specified for model {model_name} in {depth}")

            # Create a list to store all models and their names
            models_and_names = []

            for single_depth in model_depths:
                # Prepare kwargs for this depth
                kwargs = {
                    'depth': single_depth,
                    'pretrained': self.pretrained,
                    'input_channels': input_channels,
                    'num_classes': num_classes
                }

                # Special handling for models that don't use depth
                if model_name == 'yolo8resnet':
                    kwargs = {
                        'input_channels': input_channels,
                        'num_classes': num_classes,
                        'pretrained': self.pretrained
                    }

                filtered_kwargs = {k: v for k,
                                   v in kwargs.items() if k in model_params}

                # Format model name for this depth
                if model_name == 'yolo8resnet':
                    model_name_with_depth = f"{model_name}_{single_depth}"
                else:
                    model_name_with_depth = f"{model_name}_{single_depth}"

                # Create or load model
                model = self._create_or_load_model(
                    model_func, filtered_kwargs, model_name_with_depth,
                    task_name, dataset_name, adversarial
                )

                models_and_names.append((model, model_name_with_depth))

            return models_and_names
        else:
            # Single depth case
            kwargs = {
                'depth': depth,
                'pretrained': self.pretrained,
                'input_channels': input_channels,
                'num_classes': num_classes
            }

            # Special handling for models that don't use depth
            if model_name == 'yolo8resnet':
                kwargs = {
                    'input_channels': input_channels,
                    'num_classes': num_classes,
                    'pretrained': self.pretrained
                }

            filtered_kwargs = {k: v for k,
                               v in kwargs.items() if k in model_params}
            model_name_with_depth = self._format_model_name(model_name, depth)
            model = self._create_or_load_model(
                model_func, filtered_kwargs, model_name_with_depth,
                task_name, dataset_name, adversarial
            )
            # --- Ensure always returns a list of (model, model_name_with_depth) tuples ---
            # If model is a tuple or list (e.g., YOLO returns multiple heads), wrap each as (m, model_name_with_depth)
            if isinstance(model, (list, tuple)) and not isinstance(model, torch.nn.Module):
                return [(m, model_name_with_depth) for m in model]
            else:
                return [(model, model_name_with_depth)]

    def _create_or_load_model(self, model_func, kwargs, model_name_with_depth, task_name, dataset_name, adversarial=False):
        """Helper method to create or load a single model"""
        model = None
        if task_name and dataset_name:
            checkpoint_path = self.get_latest_checkpoint(
                model_name_with_depth, dataset_name, task_name, adversarial)
            if checkpoint_path:
                # Load from checkpoint
                # Load the checkpoint to CPU first to avoid OOM issues
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model = model_func(**kwargs)
                # Strip "module." prefix if present in checkpoint keys
                new_state_dict = {}
                for k, v in checkpoint.items():
                    new_key = k.replace("module.", "") if k.startswith(
                        "module.") else k
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)
                logging.info(
                    f"Loaded {'adversarial' if adversarial else 'normal'} pretrained model from checkpoint: {checkpoint_path}")
                # Free memory before moving the model to device
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if self.fp16:
                    model = model.half()  # Convert model to half precision if enabled
                model = model.to(self.device)

        if model is None:
            model = model_func(**kwargs)
            logging.info(
                f"ModelLoader: Created a new model: {model_name_with_depth}")

        # Memory optimization
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        try:
            def model_builder(): return model_func(**kwargs)
            wrapper = MemoryEfficientModel(
                model_builder, self.device, self.fp16)
            model = wrapper.load_model()
            logging.info(
                f"Successfully loaded model using memory-efficient wrapper")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Could not load model due to: {str(e)}")

        return model

    def recursive_set_param(self, model, key_parts, param):
        """Recursively set parameter in nested model structure."""
        if len(key_parts) == 1:
            if hasattr(model, key_parts[0]):
                setattr(model, key_parts[0], param)
        else:
            self.recursive_set_param(
                getattr(model, key_parts[0]), key_parts[1:], param)

    def load_pretrained_model(self, model_name, load_task, dataset_name, depth=None, input_channels=3, num_classes=None, adversarial=False):
        """Loads a pretrained model, specified by architecture, depth, and task-related information."""
        model, model_name_with_depth = self.get_model(
            model_name=model_name,
            depth=depth,
            input_channels=input_channels,
            num_classes=num_classes,
            task_name=load_task,
            dataset_name=dataset_name,
            adversarial=adversarial
        )[0]

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)  # for multi-GPU use

        # Initialize GradScaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        # ...remaining code unchanged...

        return model

    def load_multiple_models(self, model_name, depths, input_channels=3, num_classes=None, task_name=None, dataset_name=None, adversarial=False):
        """Loads multiple models of the same architecture but different depths, as specified."""
        models = {}
        for depth in depths:
            try:
                model, model_name_with_depth = self.get_model(
                    model_name=model_name,
                    depth=depth,
                    input_channels=input_channels,
                    num_classes=num_classes,
                    task_name=task_name,
                    dataset_name=dataset_name,
                    adversarial=adversarial
                )[0]
                models[depth] = model
                logging.info(
                    f"Model {model_name_with_depth} loaded successfully.")
            except ValueError as e:
                logging.error(
                    f"Failed to load model {model_name} with depth {depth}: {e}")
        return models
