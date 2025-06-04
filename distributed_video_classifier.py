import os
os.environ["FFMPEG_LOGLEVEL"] = "quiet"

import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

# For metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Try to import IPython/Jupyter specific modules
try:
    from IPython.display import display, HTML, Javascript, clear_output
    from matplotlib.gridspec import GridSpec
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# Try to import distributed modules
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class VideoClassifier:
    """
    An improved class to handle the training, validation, and testing of video classification models
    with dynamic visualization capabilities, detailed logging, and distributed training support.
    
    Supports both single GPU/CPU training (for notebooks) and multi-GPU distributed training.
    """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        base_model='resnet18',
        temporal_mode='attention',
        num_classes=3,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_dir='model_checkpoints',
        experiment_name=None,
        device=None,
        num_workers=2,
        class_weights=None,
        use_dynamic_viz=True,
        validation_freq=2,   # Number of validation runs per epoch
        viz_update_freq=20,  # Update visualization every N mini-batches
        enable_mini_validation=False,  # Enable mini-validation logic (independent of visualization)
        # Distributed training parameters
        distributed=None,    # Auto-detect if None
        local_rank=None,     # Auto-detect from environment
        world_size=None,     # Auto-detect from environment
        backend='nccl'       # Communication backend for distributed training
    ):
        """
        Initialize the video classifier with optional distributed training support
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            base_model: Base CNN architecture ('resnet18', 'resnet50', etc.)
            temporal_mode: Temporal aggregation method ('attention', 'lstm', 'gru', 'pooling', 'convolution')
            num_classes: Number of classes (3 for Normal, Near Collision, Collision)
            batch_size: Training batch size per GPU
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            save_dir: Directory to save checkpoints and results
            experiment_name: Name for this training run (for saving results)
            device: Device to train on (will use CUDA if available if None)
            num_workers: Number of workers for data loading
            class_weights: Optional tensor of class weights for handling imbalanced data
            use_dynamic_viz: Whether to use dynamic visualization (disabled in distributed mode for non-master processes)
            validation_freq: Number of validation runs per epoch
            viz_update_freq: Update visualization every N mini-batches
            enable_mini_validation: Enable mini-validation logic during training (independent of visualization)
            distributed: Whether to use distributed training (auto-detect if None)
            local_rank: Local rank for distributed training (auto-detect if None)
            world_size: World size for distributed training (auto-detect if None)
            backend: Communication backend for distributed training
        """
        
        # Auto-detect distributed training settings
        self.distributed = self._setup_distributed_training(distributed, local_rank, world_size, backend)
        
        # Set device based on distributed setup
        self.device = self._setup_device(device)
        
        # Determine if this is the master process (for logging and visualization)
        self.is_master = not self.distributed or self.local_rank == 0
        
        # Adjust batch size for distributed training
        self.original_batch_size = batch_size
        self.batch_size = batch_size  # Per GPU batch size
        
        # Save hyperparameters
        self.base_model = base_model
        self.temporal_mode = temporal_mode
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.use_dynamic_viz = use_dynamic_viz and self.is_master and JUPYTER_AVAILABLE
        self.enable_mini_validation = enable_mini_validation
        self.validation_freq = validation_freq
        self.viz_update_freq = viz_update_freq
        self.patience_counter = 0
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{base_model}_{temporal_mode}_{timestamp}"
            if self.distributed:
                self.experiment_name += f"_dist{self.world_size}gpu"
        else:
            self.experiment_name = experiment_name
        
        # Create save directory (only on master process)
        self.save_dir = os.path.join(save_dir, self.experiment_name)
        if self.is_master:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup logging (only on master process)
        if self.is_master:
            self._setup_logging()
            self._log_initialization_info()
        
        # Create data loaders with distributed sampling if needed
        self._setup_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Initialize the model
        if self.is_master:
            self.logger.info("Creating model...")
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Wrap model for distributed training if needed
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            if self.is_master:
                self.logger.info(f"Model wrapped with DistributedDataParallel")
        
        if self.is_master:
            self.logger.info("Model created and moved to device")
        
        # Set up class weights for handling imbalanced data
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            if self.is_master:
                self.logger.info(f"Class weights applied: {class_weights}")
                
        # Create loss function and optimizer
        self._setup_training()
        
        # Initialize training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'lr': []
        }
        
        # For tracking best model
        self.best_val_loss = float('inf')
        self.best_val_metrics = None
        self.best_epoch = 0
        self.best_mini_val_loss = float('inf')  # Track best mini-validation loss
        
        # For dynamic visualization
        self.current_epoch = 0
        self.visualizer = None
        
        # Validation counter
        self.validation_counter = 0
        
        if self.is_master:
            self.logger.info("VideoClassifier initialization completed successfully")
            if self.distributed:
                self.logger.info(f"Distributed training setup complete with {self.world_size} GPUs")
            self.logger.info("=" * 80)
    
    def _setup_distributed_training(self, distributed, local_rank, world_size, backend):
        """Setup distributed training if requested and available"""
        
        # Auto-detect distributed environment
        if distributed is None:
            distributed = (
                DISTRIBUTED_AVAILABLE and 
                'WORLD_SIZE' in os.environ and 
                int(os.environ.get('WORLD_SIZE', 1)) > 1
            )
        
        if not distributed:
            self.local_rank = 0
            self.world_size = 1
            return False
        
        if not DISTRIBUTED_AVAILABLE:
            print("Warning: Distributed training requested but torch.distributed not available")
            self.local_rank = 0
            self.world_size = 1
            return False
        
        # Get distributed training parameters from environment
        self.local_rank = local_rank if local_rank is not None else int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))

        os.environ.setdefault('NCCL_BLOCKING_WAIT',  '1')   # disable kill-switch
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('NCCL_TIMEOUT',        '7200')  # seconds (2h)

        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=self.world_size,
                rank=self.local_rank,
                timeout=timedelta(hours=2)
            )
        
        return True
    
    def _setup_device(self, device):
        """Setup device based on distributed training configuration"""
        if device is not None:
            return device
        
        if self.distributed:
            # In distributed mode, each process uses its assigned GPU
            device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device)
        else:
            # Single process mode - use CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return device
    
    def _setup_logging(self):
        """Setup detailed logging for training process (master process only)"""
        # Create logger
        self.logger = logging.getLogger(f'VideoClassifier_{self.experiment_name}')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create file handler
        log_file = os.path.join(self.save_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _log_initialization_info(self):
        """Log initialization information (master process only)"""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING VIDEO CLASSIFIER")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment Name: {self.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Base Model: {self.base_model}")
        self.logger.info(f"Temporal Mode: {self.temporal_mode}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Batch Size (per GPU): {self.batch_size}")
        if self.distributed:
            self.logger.info(f"Effective Batch Size (total): {self.batch_size * self.world_size}")
            self.logger.info(f"World Size: {self.world_size}")
            self.logger.info(f"Local Rank: {self.local_rank}")
        self.logger.info(f"Learning Rate: {self.learning_rate}")
        self.logger.info(f"Weight Decay: {self.weight_decay}")
        self.logger.info(f"Validation Frequency: {self.validation_freq} times per epoch")
        self.logger.info(f"Mini-validation Enabled: {self.enable_mini_validation}")
        self.logger.info(f"Dynamic Visualization Enabled: {self.use_dynamic_viz}")
    
    def _setup_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """Setup data loaders with distributed sampling if needed"""
        
        # Create distributed samplers if using distributed training
        if self.distributed:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
            self.test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=(self.train_sampler is None), 
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        if self.is_master:
            self.logger.info(f"Data Loaders Created:")
            self.logger.info(f"  Train batches: {len(self.train_loader)}")
            self.logger.info(f"  Validation batches: {len(self.val_loader)}")
            self.logger.info(f"  Test batches: {len(self.test_loader)}")
            if self.distributed:
                self.logger.info(f"  Using DistributedSampler for all loaders")
    
    def _create_model(self):
        """Create the model with the correct output layer for multi-class classification"""
        from nexar_arch import EnhancedFrameCNN
        
        # CRITICAL: Set seed before model creation to ensure identical initialization
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create the base model with a scalar output
        model = EnhancedFrameCNN(
            base_model=self.base_model,
            pretrained=True,
            dropout_rate=0.5,
            temporal_mode=self.temporal_mode,
            store_attention_weights=True
        )
        
        # Replace the final linear layer to match the number of classes
        if self.num_classes > 2:
            # Get the feature dimension from the original classifier
            feature_dim = model.classifier[-1].in_features
            
            # IMPORTANT: Keep same seed for final layer creation
            torch.manual_seed(42)  # Same seed as before!
            # Replace the final layer with one that outputs the correct number of classes
            model.classifier[-1] = nn.Linear(feature_dim, self.num_classes)
            if self.is_master:
                self.logger.info(f"Replaced final layer: {feature_dim} -> {self.num_classes} classes")
        
        # Reset seed after everything is done
        torch.manual_seed(int(time.time()))
        
        return model
    
    def _setup_training(self):
        """Set up loss function, optimizer, and scheduler"""
        # Loss function depends on number of classes
        if self.num_classes == 2:
            # Binary classification with class weights if provided
            if self.class_weights is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                if self.is_master:
                    self.logger.info("Using BCEWithLogitsLoss with class weights")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                if self.is_master:
                    self.logger.info("Using BCEWithLogitsLoss")
        else:
            # Multi-class classification with class weights if provided
            if self.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                if self.is_master:
                    self.logger.info("Using CrossEntropyLoss with class weights")
            else:
                self.criterion = nn.CrossEntropyLoss()
                if self.is_master:
                    self.logger.info("Using CrossEntropyLoss")
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        if self.is_master:
            self.logger.info(f"Optimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        
        # Learning rate scheduler - Cosine annealing works well for deep learning
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=30,  # Maximum number of iterations
            eta_min=self.learning_rate/100  # Minimum learning rate
        )
        if self.is_master:
            self.logger.info(f"Scheduler: CosineAnnealingLR (T_max=30, eta_min={self.learning_rate/100})")
    
    def _init_visualizer(self, epochs):
        """Initialize the dynamic visualizer with proper parameters (master process only)"""
        if not self.use_dynamic_viz or not self.is_master:
            if self.is_master:
                self.logger.info("Dynamic visualization disabled")
            return
        
        try:
            from dynamic_training_visualizer import DynamicTrainingVisualizer
            
            if self.is_master:
                self.logger.info("Initializing dynamic visualizer...")
            
            # Create visualizer
            self.visualizer = DynamicTrainingVisualizer(
                num_epochs=epochs,
                num_iterations_per_epoch=len(self.train_loader),
                update_freq=self.viz_update_freq,
                num_classes=self.num_classes,
                class_names=['Normal', 'Near Collision', 'Collision'] if self.num_classes == 3 else None
            )
            
            # Initialize display
            self.visualizer.initialize_display()
            if self.is_master:
                self.logger.info("Dynamic visualizer initialized successfully")
        except ImportError:
            if self.is_master:
                self.logger.warning("Dynamic visualizer not available, continuing without visualization")
            self.visualizer = None
    
    def _all_reduce_tensor(self, tensor):
        """All-reduce a tensor across all processes in distributed training"""
        if not self.distributed:
            return tensor
            
        # Clone tensor to avoid modifying original
        reduced_tensor = tensor.clone()
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        reduced_tensor /= self.world_size
        return reduced_tensor
    
    def _gather_metrics_from_all_processes(self, local_metrics):
        """Gather and aggregate metrics from all processes"""
        if not self.distributed:
            return local_metrics
        
        # Convert metrics to tensors for all_reduce
        aggregated_metrics = {}
        
        for key, value in local_metrics.items():
            if isinstance(value, dict):
                # Per-class metrics
                aggregated_metrics[key] = {}
                for cls_idx, cls_value in value.items():
                    tensor_value = torch.tensor(cls_value, device=self.device)
                    aggregated_value = self._all_reduce_tensor(tensor_value)
                    aggregated_metrics[key][cls_idx] = aggregated_value.item()
            else:
                # Scalar metrics
                tensor_value = torch.tensor(value, device=self.device)
                aggregated_value = self._all_reduce_tensor(tensor_value)
                aggregated_metrics[key] = aggregated_value.item()
        
        return aggregated_metrics
            
    def train(self, epochs=30, patience=5, mixed_precision=True):
        """
        Train with async validation - same pattern as mini-batch validation
        """
        if self.is_master:
            self.logger.info("=" * 80)
            self.logger.info("STARTING TRAINING")
            self.logger.info("=" * 80)
        
        # Setup...
        if mixed_precision and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        else:
            scaler = None
        
        if self.use_dynamic_viz and self.is_master:
            self._init_visualizer(epochs)
        
        patience_counter = 0
        
        # Main training loop - NO BLOCKING VALIDATION
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Set epoch for distributed sampler - all processes do this together
            if self.distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            if self.is_master:
                self.logger.info("=" * 60)
                self.logger.info(f"EPOCH {epoch+1}/{epochs}")
                self.logger.info("=" * 60)
            
            # Update visualizer (master only)
            if self.visualizer and self.is_master:
                self.visualizer.start_epoch(epoch + 1)
            
            # Train one epoch - all processes participate
            if self.is_master:
                self.logger.info("Starting training phase...")
            train_loss = self._train_epoch(scaler)
            if self.is_master:
                self.logger.info(f"Training phase completed. Average train loss: {train_loss:.6f}")
            
            # KEY CHANGE: Validation happens ASYNC, doesn't block next epoch
            # Schedule validation for master process but don't wait for it
            if self.is_master:
                # Start validation asynchronously (like mini-validation)
                self._schedule_end_epoch_validation(epoch, train_loss, start_time)
            
            # Update learning rate - all processes
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            if self.is_master:
                self.logger.info(f"Learning rate updated: {old_lr:.8f} -> {new_lr:.8f}")
            
            # Early stopping check - use previous epoch's results
            should_stop = False
            if self.is_master and epoch > 0:  # Skip first epoch
                should_stop = patience_counter >= patience
            
            # Broadcast early stopping decision
            if self.distributed:
                if self.is_master:
                    stop_tensor = torch.tensor(1 if should_stop else 0, device=self.device, dtype=torch.int)
                else:
                    stop_tensor = torch.tensor(0, device=self.device, dtype=torch.int)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1
            
            # Mark end of epoch in visualizer (master only)
            if self.visualizer and self.is_master:
                current_iteration = (epoch + 1) * len(self.train_loader)
                self.visualizer.mark_epoch(epoch + 1, current_iteration)
            
            # Set model back to training mode
            self.model.train()
            
            # Early stopping
            if should_stop:
                if self.is_master:
                    self.logger.info("Early stopping triggered")
                break
        
        # Wait for any pending validation to complete
        if self.is_master:
            self._finalize_validation_and_training()
        
        return self.history if self.is_master else None
    
    
    def _schedule_end_epoch_validation(self, epoch, train_loss, start_time):
        """
        Handle end-of-epoch validation without blocking other processes
        Similar to mini-validation pattern
        """
        if not self.is_master:
            return
        
        self.logger.info("Starting end-of-epoch validation (non-blocking)...")
        
        # Run validation (only master, others continue)
        val_metrics = self._validate(detailed=False)
        
        self.logger.info("End-of-epoch validation completed")
        self._log_validation_results(val_metrics)
        
        # Update visualizer
        if self.visualizer:
            current_iteration = (epoch + 1) * len(self.train_loader)
            self.visualizer.update_full_val_metrics(current_iteration, val_metrics['loss'], val_metrics)
        
        # Save epoch checkpoint
        self.logger.info("Saving epoch checkpoint...")
        self._save_epoch_checkpoint(epoch + 1, val_metrics)
        
        # Check for best model and update patience counter
        improved = self._check_and_save_best_model(epoch, val_metrics)
        if improved:
            self.patience_counter = 0  # Store as instance variable
            self.logger.info("*** NEW BEST MODEL FOUND ***")
        else:
            self.patience_counter += 1
            self.logger.info(f"No improvement. Patience counter: {self.patience_counter}/{5}")
        
        # Update history
        self._update_history(epoch, train_loss, val_metrics, self.optimizer.param_groups[0]['lr'])
        
        # Print progress
        self._log_epoch_summary(epoch, start_time, train_loss, val_metrics)
    
    
    def _finalize_validation_and_training(self):
        """
        Complete any pending validation and finalize training
        """
        if not self.is_master:
            return
        
        # Final validation for the last epoch if needed
        self.logger.info("Finalizing training...")
        
        # Save training history
        self._save_history()
        
        # Load best model
        if os.path.exists(os.path.join(self.save_dir, 'best_model.pth')):
            self.logger.info(f"Loading best model from epoch {self.best_epoch}...")
            self._load_checkpoint('best_model.pth')
        
        # Plot training history
        self._plot_training_history()
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        if self.best_val_metrics:
            self.logger.info(f"Best validation accuracy: {self.best_val_metrics['accuracy']:.4f}")
        self.logger.info("=" * 80)
    
        
    def _train_epoch(self, scaler=None):
        """Train for one epoch with dynamic visualization and distributed support"""
        # Make sure model is in training mode
        self.model.train()
        
        train_loss = 0.0
        total_samples = 0
        
        # Training progress bar (only on master process)
        if self.is_master:
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        else:
            train_pbar = self.train_loader
        
        # Calculate validation frequency
        val_step = max(1, len(self.train_loader) // self.validation_freq)
        if self.is_master:
            self.logger.info(f"Will run mini-validation every {val_step} batches")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Get data
            frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
        targets = batch['target']
        
        # Convert string targets to numeric if needed
        if isinstance(targets[0], str):
            class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
            targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
        else:
            targets = targets.to(self.device)
        
        # Forward pass with mixed precision if available
        self.optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = self.model(frames)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        
        # Update metrics
        current_loss = loss.item()
        batch_size = len(targets)
        train_loss += current_loss * batch_size
        total_samples += batch_size
        
        # Update progress bar (master process only)
        if self.is_master:
            avg_loss = train_loss / total_samples
            train_pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        # Mini-validation logic (EXACTLY LIKE THIS WORKS PERFECTLY)
        if (self.enable_mini_validation and 
            (batch_idx + 1) % val_step == 0 and 
            batch_idx > 0):
            
            if self.is_master:
                current_iteration = self.current_epoch * len(self.train_loader) + batch_idx + 1
                self._handle_mini_validation(batch_idx, current_iteration)
        
        # Free memory
        del frames, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        # Calculate average loss - NO SYNCHRONIZATION (exactly like mini-validation)
        avg_train_loss = train_loss / total_samples
        
        return avg_train_loss
        
    def _handle_mini_validation(self, batch_idx, current_iteration):
        """Handle mini-validation during training (master process only, runs regardless of visualization)"""
        if not self.is_master:
            return
            
        self.logger.info(f"Running mini-validation at batch {batch_idx+1}/{len(self.train_loader)}")
        
        # Run mini-validation with fewer batches for speed
        val_metrics = self._mini_validate()
        
        self.logger.info(f"Mini-validation results:")
        self.logger.info(f"  Loss: {val_metrics['loss']:.6f}")
        self.logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        self.logger.info(f"  Based on {min(25, len(self.val_loader))} batches")
        
        # Update visualization with validation results (only if visualization is enabled)
        if self.use_dynamic_viz and self.visualizer:
            self.visualizer.update_val_metrics(current_iteration, val_metrics['loss'], val_metrics)
        
        # Check if this is the best mini-validation loss (this logic always runs on master)
        if val_metrics['loss'] < self.best_mini_val_loss:
            improvement = self.best_mini_val_loss - val_metrics['loss']
            self.best_mini_val_loss = val_metrics['loss']
            
            self.logger.info(f"*** NEW BEST MINI-VALIDATION LOSS ***")
            self.logger.info(f"Improvement: {improvement:.6f}")
            
            # Run full validation
            self.logger.info("Running full validation due to mini-validation improvement...")
            full_val_metrics = self._validate(detailed=False)
            
            self.logger.info(f"Full validation results:")
            self.logger.info(f"  Loss: {full_val_metrics['loss']:.6f}")
            self.logger.info(f"  Accuracy: {full_val_metrics['accuracy']:.4f}")
            
            # Update visualizer with full validation results (only if visualization is enabled)
            if self.use_dynamic_viz and self.visualizer:
                self.visualizer.update_full_val_metrics(current_iteration, full_val_metrics['loss'], full_val_metrics)
            
            # Check if this is the best full validation result (this logic always runs)
            if full_val_metrics['loss'] < self.best_val_loss:
                full_improvement = self.best_val_loss - full_val_metrics['loss']
                self.best_val_loss = full_val_metrics['loss']
                self.best_val_metrics = full_val_metrics
                self.best_epoch = self.current_epoch + 1
                
                # Save best model
                self._save_checkpoint('best_model.pth')
                self._save_validation_results(f'best_validation_epoch{self.current_epoch+1}_iter{current_iteration}', full_val_metrics)
                
                self.logger.info(f"*** NEW BEST FULL VALIDATION LOSS ***")
                self.logger.info(f"Improvement: {full_improvement:.6f}")
                self.logger.info("Best model saved")
        
        # IMPORTANT: Set model back to training mode after validation
        self.model.train()
        
    def _validate(self, detailed: bool = True, log_every: int = 50) -> dict:
        """
        Validation function - EXACTLY SAME LOGIC AS MINI-VALIDATION
        * Only rank-0 actually runs validation
        * All other ranks return immediately with empty dict
        * NO synchronization, NO barriers - exactly like _mini_validate works
        """
    
        # EXACTLY SAME AS MINI-VALIDATION: skip fast for non-master ranks
        if self.distributed and not self.is_master:
            return {}
    
        import time
        t0 = time.time()
        self.logger.info("[VAL] starting full validation…")
    
        # Create a clean DataLoader (same as mini-validation)
        val_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            timeout=60
        )
    
        # Set model to eval mode (same as mini-validation)
        self.model.eval()
        total_samples, correct, total_loss = 0, 0, 0.0
        all_preds, all_targets = [], []
    
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                try:
                    frames = batch['frames'].permute(0, 4, 1, 2, 3).float()\
                             .to(self.device, non_blocking=True)
                    targets = batch['target']
                    if isinstance(targets[0], str):
                        class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                        targets = torch.tensor([class_map[t] for t in targets],
                                               device=self.device)
                    else:
                        targets = targets.to(self.device, non_blocking=True)
    
                    outputs = self.model(frames)
                    loss    = self.criterion(outputs, targets)
    
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == targets).sum().item()
                    total_loss += loss.item() * targets.size(0)
                    total_samples += targets.size(0)
    
                    # ALWAYS collect predictions and targets (not dependent on detailed flag)
                    all_preds   += preds.cpu().tolist()
                    all_targets += targets.cpu().tolist()
    
                    if (i % log_every) == 0:
                        self.logger.info(f"[VAL] batch {i}/{len(val_loader)} "
                                         f"loss={loss.item():.4f}")
    
                except Exception as e:
                    self.logger.warning(f"[VAL] batch {i} failed: {e}")
                    continue
    
        # Calculate final metrics (only rank-0)
        if total_samples == 0:
            avg_loss, acc = float('inf'), 0.0
            metrics = {'loss': avg_loss, 'accuracy': acc, 'total_samples': total_samples}
        else:
            avg_loss = total_loss / total_samples
            acc = correct / total_samples
            metrics = {'loss': avg_loss, 'accuracy': acc, 'total_samples': total_samples}
            
            # Calculate detailed metrics using the collected predictions
            detailed_metrics = self._calculate_metrics(all_preds, all_targets, detailed=detailed)
            metrics.update(detailed_metrics)
    
        dt = time.time() - t0
        self.logger.info(f"[VAL] done in {dt:.1f}s | loss={avg_loss:.4f} "
                         f"| acc={acc:.4f}")
    
        # IMPORTANT: DO NOT set model back to train() here!
        # Let the calling function handle it (same as mini-validation)
        
        return metrics
        
    def _mini_validate(self, max_batches=25):
        """
        Run a smaller validation for dynamic updates during training - FIXED
        """
        if not self.is_master:
            return {}
        
        # Create a clean DataLoader for mini-validation (same approach as full validation)
        if not hasattr(self, '_mini_val_loader'):
            from torch.utils.data import DataLoader
            
            if self.distributed:
                # Create clean mini-validation loader WITHOUT DistributedSampler
                self._mini_val_loader = DataLoader(
                    self.val_loader.dataset, 
                    batch_size=self.batch_size,  # Use self.batch_size instead of self.val_loader.batch_size
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    sampler=None,  # No distributed sampler!
                    drop_last=False
                )
            else:
                # Single GPU case
                self._mini_val_loader = DataLoader(
                    self.val_loader.dataset, 
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=False
                )
        
        # Switch to eval mode
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        batches_processed = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self._mini_val_loader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # Get data
                    frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device, non_blocking=True)
                    targets = batch['target']
                    
                    # Convert string targets to numeric if needed
                    if isinstance(targets[0], str):
                        class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                        targets = torch.tensor([class_map[t] for t in targets]).to(self.device, non_blocking=True)
                    else:
                        targets = targets.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, targets)
                    
                    # Store results - move to CPU immediately (mini-validation is master-only)
                    val_loss += loss.item() * len(targets)
                    all_outputs.append(outputs.cpu())
                    all_targets.append(targets.cpu())
                    batches_processed += 1
                    total_samples += len(targets)
                    
                    # Free memory
                    del frames, targets, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.warning(f"Error in mini-validation batch: {e}")
                    continue
        
        # Calculate metrics on the subset
        if total_samples > 0:
            val_loss /= total_samples
            
            # Concatenate results
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_outputs, all_targets, detailed=False)
            metrics['loss'] = val_loss
            metrics['batches_used'] = batches_processed
            metrics['samples_used'] = total_samples
            # Fix the percentage calculation - use the actual dataset size
            total_val_samples = len(self.val_loader.dataset)
            metrics['percentage_of_val_set'] = (total_samples / total_val_samples) * 100
        else:
            metrics = {'loss': float('inf'), 'accuracy': 0.0}
        
        return metrics
    
    def _calculate_metrics(self, outputs_or_preds, targets, detailed=False):
        """
        Unified metrics calculation function.
        
        Args:
            outputs_or_preds: Either torch tensors (model outputs) or lists/arrays (predictions)
            targets: Either torch tensors or lists/arrays (ground truth)
            detailed: If True, include confusion matrix and classification report
        
        Returns:
            Dictionary with calculated metrics
        """
        if len(targets) == 0:
            return {'accuracy': 0.0, 'precision': {}, 'recall': {}, 'f1': {}}
        
        try:
            import numpy as np
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                precision_recall_fscore_support, confusion_matrix, 
                classification_report, roc_auc_score
            )
            
            # Handle different input types
            if isinstance(outputs_or_preds, torch.Tensor) and outputs_or_preds.dim() > 1:
                # Case 1: Model outputs (torch tensors) - convert to predictions
                if self.num_classes == 2:
                    y_pred = (torch.sigmoid(outputs_or_preds) > 0.5).float().numpy()
                    y_prob = torch.sigmoid(outputs_or_preds).numpy()
                else:
                    y_pred = torch.argmax(outputs_or_preds, dim=1).numpy()
                    y_prob = torch.softmax(outputs_or_preds, dim=1).numpy()
                
                y_true = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
                
            else:
                # Case 2: Already predictions (lists/arrays)
                y_pred = np.array(outputs_or_preds)
                y_true = np.array(targets)
                y_prob = None  # No probabilities available
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Per-class metrics using sklearn
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Convert to dictionaries
            per_class_precision = {i: float(precision[i]) for i in range(len(precision))}
            per_class_recall = {i: float(recall[i]) for i in range(len(recall))}
            per_class_f1 = {i: float(f1[i]) for i in range(len(f1))}
            
            # Weighted averages
            weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Build metrics dictionary
            metrics = {
                'accuracy': float(accuracy),
                'precision': per_class_precision,
                'recall': per_class_recall,
                'f1': per_class_f1,
                'weighted_precision': float(weighted_precision),
                'weighted_recall': float(weighted_recall),
                'weighted_f1': float(weighted_f1),
                'samples_used': len(y_pred)
            }
            
            # Add AUC if probabilities are available
            if y_prob is not None:
                try:
                    if self.num_classes == 2:
                        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
                    else:
                        # One-vs-rest AUC for multi-class
                        metrics['auc'] = float(roc_auc_score(
                            np.eye(self.num_classes)[y_true],  # Convert to one-hot
                            y_prob, 
                            multi_class='ovr',
                            average='weighted'
                        ))
                except Exception as e:
                    if self.is_master:
                        self.logger.warning(f"Error calculating AUC: {e}")
                    metrics['auc'] = 0.5
            
            # Add detailed metrics if requested
            if detailed:
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    
                    metrics['confusion_matrix'] = cm.tolist()
                    metrics['classification_report'] = class_report
                except Exception as e:
                    if self.is_master:
                        self.logger.warning(f"Error calculating detailed metrics: {e}")
            
            return metrics
            
        except Exception as e:
            # Fallback to basic metrics if sklearn fails
            if self.is_master:
                self.logger.error(f"Failed to calculate metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': {},
                'recall': {},
                'f1': {},
                'samples_used': len(targets) if hasattr(targets, '__len__') else 0
            }
    
    def _log_validation_results(self, val_metrics):
        """Log validation results (master process only)"""
        if not self.is_master:
            return
            
        self.logger.info(f"Validation Results:")
        self.logger.info(f"  Loss: {val_metrics['loss']:.6f}")
        self.logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Log per-class metrics
        if isinstance(val_metrics['precision'], dict):
            self.logger.info("Per-class metrics:")
            class_names = ['Normal', 'Near Collision', 'Collision']
            for cls in range(self.num_classes):
                class_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                prec = val_metrics['precision'].get(cls, 0)
                rec = val_metrics['recall'].get(cls, 0)
                f1 = val_metrics['f1'].get(cls, 0)
                self.logger.info(f"    {class_name}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        
        if 'auc' in val_metrics:
            self.logger.info(f"  AUC: {val_metrics['auc']:.4f}")
    
    def _update_history(self, epoch, train_loss, val_metrics, new_lr):
        """Update training history (master process only)"""
        if not self.is_master:
            return
            
        self.history['epoch'].append(epoch + 1)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        
        # For metrics that are dictionaries (per-class metrics), store the average
        for key in ['precision', 'recall', 'f1']:
            if key in val_metrics and isinstance(val_metrics[key], dict):
                avg_value = np.mean(list(val_metrics[key].values()))
                self.history[f'val_{key}'].append(avg_value)
            
        if 'auc' in val_metrics:
            self.history['val_auc'].append(val_metrics['auc'])
            
        self.history['lr'].append(new_lr)
    
    def _check_and_save_best_model(self, epoch, val_metrics):
        """Check for best model and save if improved (master process only)"""
        if not self.is_master:
            return False
            
        self.logger.info("Checking for best model...")
        self.logger.info(f"Current validation loss: {val_metrics['loss']:.6f}")
        self.logger.info(f"Best validation loss so far: {self.best_val_loss:.6f}")
        
        if val_metrics['loss'] < self.best_val_loss:
            improvement = self.best_val_loss - val_metrics['loss']
            self.best_val_loss = val_metrics['loss']
            self.best_val_metrics = val_metrics
            self.best_epoch = epoch + 1
            
            # Save best model
            self._save_checkpoint('best_model.pth')
            self.logger.info(f"*** NEW BEST MODEL FOUND ***")
            self.logger.info(f"Improvement: {improvement:.6f}")
            self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
            self.logger.info(f"Best epoch updated to: {self.best_epoch}")
            self.logger.info("Best model saved")
            return True
        return False
    
    def _log_epoch_summary(self, epoch, start_time, train_loss, val_metrics):
        """Log epoch summary (master process only)"""
        if not self.is_master:
            return
            
        epoch_time = time.time() - start_time
        self.logger.info(f"Epoch Summary:")
        self.logger.info(f"  Time: {epoch_time:.2f}s")
        self.logger.info(f"  Train Loss: {train_loss:.6f}")
        self.logger.info(f"  Val Loss: {val_metrics['loss']:.6f}")
        self.logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        if isinstance(val_metrics.get('f1'), dict):
            avg_f1 = np.mean(list(val_metrics['f1'].values()))
            self.logger.info(f"  Val F1 (avg): {avg_f1:.4f}")
    
    def _finalize_training(self):
        """Finalize training process (master process only)"""
        if not self.is_master:
            return
            
        # Save training history
        self.logger.info("Saving training history...")
        self._save_history()
        
        # Load best model
        self.logger.info(f"Loading best model from epoch {self.best_epoch}...")
        self._load_checkpoint('best_model.pth')
        
        # Plot training history
        self.logger.info("Plotting training history...")
        self._plot_training_history()
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Best epoch: {self.best_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        if self.best_val_metrics:
            self.logger.info(f"Best validation accuracy: {self.best_val_metrics['accuracy']:.4f}")
        self.logger.info("=" * 80)
    
    def _save_epoch_checkpoint(self, epoch, val_metrics):
        """Save checkpoint for each epoch (master process only)"""
        if not self.is_master:
            return
            
        checkpoint_name = f'checkpoint_epoch{epoch}.pth'
        self._save_checkpoint(checkpoint_name)
        self._save_validation_results(f'validation_epoch{epoch}', val_metrics)
    
    def _save_validation_results(self, filename, metrics):
        """Save validation results to JSON file (master process only)"""
        if not self.is_master:
            return
            
        results = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.current_epoch + 1,
            'metrics': {}
        }
        
        # Convert metrics to serializable format
        for key, value in metrics.items():
            if isinstance(value, dict):
                results['metrics'][key] = {str(k): float(v) for k, v in value.items()}
            else:
                results['metrics'][key] = float(value)
        
        # Save to JSON
        json_path = os.path.join(self.save_dir, f'{filename}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def test(self):
        """Test the model on the test set with distributed support"""
        if self.is_master:
            self.logger.info("=" * 80)
            self.logger.info("STARTING TESTING")
            self.logger.info("=" * 80)
        
        self.model.eval()
        test_loss = 0.0
        all_targets = []
        all_outputs = []
        all_ids = []
        
        batches_processed = 0
        total_samples = 0
        
        with torch.no_grad():
            # Use tqdm only on master process
            test_loader = tqdm(self.test_loader, desc="Testing") if self.is_master else self.test_loader
            
            for batch in test_loader:
                # Get data
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
                targets = batch['target']
                ids = batch['id'] if 'id' in batch else [f"sample_{i}" for i in range(len(targets))]
                
                # Convert string targets to numeric if needed
                if isinstance(targets[0], str):
                    class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)
                
                # Store results
                batch_loss = loss.item() * len(targets)
                test_loss += batch_loss
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_ids.extend(ids)
                batches_processed += 1
                total_samples += len(targets)
                
                # Free memory
                del frames, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate results from this process
        if total_samples > 0:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        else:
            all_outputs = torch.empty(0, self.num_classes)
            all_targets = torch.empty(0, dtype=torch.long)
        
        if self.distributed:
            # Gather results from all processes
            gathered_outputs = [torch.zeros_like(all_outputs) for _ in range(self.world_size)]
            gathered_targets = [torch.zeros_like(all_targets) for _ in range(self.world_size)]
            
            # Gather outputs and targets
            dist.all_gather(gathered_outputs, all_outputs)
            dist.all_gather(gathered_targets, all_targets)
            
            # Concatenate all results
            all_outputs = torch.cat(gathered_outputs, dim=0)
            all_targets = torch.cat(gathered_targets, dim=0)
            
            # Aggregate loss
            loss_tensor = torch.tensor(test_loss, device=self.device)
            samples_tensor = torch.tensor(total_samples, device=self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            
            test_loss = loss_tensor.item()
            total_samples = samples_tensor.item()
            
            # For IDs, we need to gather them differently (only on master process)
            if self.is_master:
                # Gather all IDs from all processes
                gathered_ids = [None for _ in range(self.world_size)]
                dist.gather_object(all_ids, gathered_ids if self.local_rank == 0 else None, dst=0)
                if gathered_ids[0] is not None:
                    all_ids = [id for ids_list in gathered_ids for id in ids_list]
            else:
                dist.gather_object(all_ids, None, dst=0)
                all_ids = []
        
        # Calculate metrics (only on master process for logging)
        if total_samples > 0:
            test_loss /= total_samples
            metrics = self._calculate_metrics(all_outputs, all_targets, detailed=False)
            metrics['loss'] = test_loss
        else:
            metrics = {'loss': float('inf'), 'accuracy': 0.0}
        
        # Save and log results (master process only)
        if self.is_master:
            self._save_validation_results('test_results', metrics)
            
            # Print results
            self.logger.info("Test Results:")
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key} per class:")
                    for cls, val in value.items():
                        class_name = ['Normal', 'Near Collision', 'Collision'][cls]
                        self.logger.info(f"    {class_name}: {val:.4f}")
                    
            # Generate confusion matrix
            self._plot_confusion_matrix(all_outputs, all_targets)
            
            # Save predictions
            self._save_predictions(all_outputs, all_targets, all_ids)
            
            self.logger.info("=" * 80)
            self.logger.info("TESTING COMPLETED")
            self.logger.info("=" * 80)
        
        return metrics if self.is_master else None
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint (master process only)"""
        if not self.is_master:
            return
            
        # Get the actual model (unwrap DDP if needed)
        model_to_save = self.model.module if self.distributed else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metrics': self.best_val_metrics,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'hyperparams': {
                'base_model': self.base_model,
                'temporal_mode': self.temporal_mode,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'distributed': self.distributed,
                'world_size': self.world_size if self.distributed else 1,
            }
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def _load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        
        # Get the actual model (unwrap DDP if needed)
        model_to_load = self.model.module if self.distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.best_epoch = checkpoint['best_epoch']
        
        if self.is_master:
            self.logger.info(f"Loaded model from epoch {self.best_epoch} with validation loss {self.best_val_loss:.6f}")
    
    def _save_history(self):
        """Save training history to CSV (master process only)"""
        if not self.is_master:
            return
            
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        self.logger.info("Training history saved to CSV")
    
    def _plot_training_history(self):
        """Plot training history graphs at the end of training (master process only)"""
        if not self.is_master:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation metrics
        plt.subplot(2, 2, 2)
        plt.plot(self.history['epoch'], self.history['val_accuracy'], label='Accuracy')
        if 'val_f1' in self.history:
            plt.plot(self.history['epoch'], self.history['val_f1'], label='F1 Score')
        if 'val_auc' in self.history:
            plt.plot(self.history['epoch'], self.history['val_auc'], label='AUC')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        # Plot precision and recall
        plt.subplot(2, 2, 3)
        if 'val_precision' in self.history:
            plt.plot(self.history['epoch'], self.history['val_precision'], label='Precision')
        if 'val_recall' in self.history:
            plt.plot(self.history['epoch'], self.history['val_recall'], label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 4)
        plt.plot(self.history['epoch'], self.history['lr'])
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
        self.logger.info("Training history plots saved")
    
    def _plot_confusion_matrix(self, outputs, targets):
        """Plot confusion matrix (master process only)"""
        if not self.is_master:
            return
            
        # Get predictions
        if self.num_classes == 2:
            # Binary classification
            y_pred = (torch.sigmoid(outputs) > 0.5).float().numpy()
        else:
            # Multi-class classification
            y_pred = torch.argmax(outputs, dim=1).numpy()
        
        y_true = targets.numpy()
        
        # Class labels
        if self.num_classes == 2:
            class_names = ['Negative', 'Positive']
        else:
            class_names = ['Normal', 'Near Collision', 'Collision']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()
        self.logger.info("Confusion matrix saved")
    
    def _save_predictions(self, outputs, targets, ids):
        """Save model predictions to CSV (master process only)"""
        if not self.is_master:
            return
            
        # Convert outputs to predictions and probabilities
        if self.num_classes == 2:
            # Binary classification
            probs = torch.sigmoid(outputs).numpy()
            preds = (probs > 0.5).astype(int)
            
            # Create DataFrame
            results_df = pd.DataFrame({
                'id': ids,
                'true_label': targets.numpy(),
                'predicted_label': preds,
                'probability': probs
            })
        else:
            # Multi-class classification
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = torch.argmax(outputs, dim=1).numpy()
            
            # Create DataFrame
            results_df = pd.DataFrame({
                'id': ids,
                'true_label': targets.numpy(),
                'predicted_label': preds
            })
            
            # Add probability columns for each class
            for i in range(self.num_classes):
                class_name = ['Normal', 'Near Collision', 'Collision'][i]
                results_df[f'prob_{class_name}'] = probs[:, i]
        
        # Save to CSV
        results_df.to_csv(os.path.join(self.save_dir, 'test_predictions.csv'), index=False)
        self.logger.info(f"Predictions saved to test_predictions.csv")
        
        return results_df
    
    def visualize_predictions(self, num_samples=5):
        """
        Visualize model predictions on random samples from the test set
        Only runs on master process.
        
        Args:
            num_samples: Number of samples to visualize
        """
        if not self.is_master:
            return
            
        self.logger.info(f"Creating prediction visualization for {num_samples} samples")
        
        # Create DataLoader with batch size 1
        loader = DataLoader(self.test_loader.dataset, batch_size=1, shuffle=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create figure for visualizations
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Class labels
        class_map = {0: 'Normal', 1: 'Near Collision', 2: 'Collision'}
        class_colors = {
            'Normal': 'green',
            'Near Collision': 'orange',
            'Collision': 'red'
        }
        
        # Sample from dataset
        samples_processed = 0
        with torch.no_grad():
            for batch in loader:
                if samples_processed >= num_samples:
                    break
                    
                # Get data
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
                targets = batch['target']
                
                # Convert string targets to numeric if needed
                if isinstance(targets[0], str):
                    # Map class names to indices
                    class_map_rev = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    target_idx = class_map_rev[targets[0]]
                else:
                    target_idx = targets[0].item()
                    
                target_class = class_map[target_idx]
                
                # Forward pass
                outputs = self.model(frames)
                
                # Get predictions
                if self.num_classes == 2:
                    # Binary classification
                    prob = torch.sigmoid(outputs)[0].item()
                    pred_idx = 1 if prob > 0.5 else 0
                    probs = [1 - prob, prob]
                else:
                    # Multi-class classification
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    pred_idx = torch.argmax(outputs, dim=1)[0].item()
                
                pred_class = class_map[pred_idx]
                
                # Get middle frame from the video for visualization
                middle_frame_idx = frames.shape[2] // 2
                middle_frame = frames[0, :, middle_frame_idx].permute(1, 2, 0).cpu().numpy()
                
                # Denormalize the frame if needed
                if middle_frame.max() <= 1.0:
                    # Denormalize using ImageNet mean and std
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    middle_frame = middle_frame * std + mean
                    middle_frame = np.clip(middle_frame, 0, 1)
                
                # Plot the frame
                ax = axes[samples_processed]
                ax.imshow(middle_frame)
                
                # Add prediction information as title
                title = f"True: {target_class} (Pred: {pred_class})"
                ax.set_title(title, color='green' if pred_class == target_class else 'red')
                
                # Add a bar chart of class probabilities
                # Create an inset axis for the probability bars
                inset_ax = ax.inset_axes([0.05, 0.05, 0.4, 0.2])
                classes = list(class_map.values())
                inset_ax.bar(classes, probs, color=[class_colors[c] for c in classes])
                inset_ax.set_ylim(0, 1)
                inset_ax.set_ylabel('Probability')
                for tick in inset_ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
                
                # Remove axes for the main image for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                samples_processed += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'prediction_visualization.png'))
        plt.close()
        
        self.logger.info("Prediction visualization saved")
    
    def visualize_attention_weights(self):
        """
        Visualize attention weights for models with attention-based temporal aggregation
        Only runs on master process.
        """
        if not self.is_master:
            return
            
        # Check if model uses attention
        model_to_check = self.model.module if self.distributed else self.model
        if (self.temporal_mode != 'attention' or 
            not hasattr(model_to_check, 'temporal_aggregation') or 
            not hasattr(model_to_check.temporal_aggregation, 'get_attention_weights')):
            self.logger.warning("Model does not support attention visualization")
            return
        
        self.logger.info("Creating attention weight visualization")
        
        # Create DataLoader with batch size 1
        loader = DataLoader(self.test_loader.dataset, batch_size=1, shuffle=True)
        
        # Get a sample
        batch = next(iter(loader))
        
        # Get data
        frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
        targets = batch['target']
        
        # Convert string targets to numeric if needed
        if isinstance(targets[0], str):
            # Map class names to indices
            class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
            target_idx = class_map[targets[0]]
        else:
            target_idx = targets[0].item()
        
        # Forward pass to get attention weights
        with torch.no_grad():
            _ = self.model(frames)
            attn_weights = model_to_check.temporal_aggregation.last_attn_weights
        
        if attn_weights is None:
            self.logger.warning("No attention weights available")
            return
        
        # Get frame sequence from the video
        # Convert from [B, C, T, H, W] back to [T, H, W, C] for display
        video_frames = frames[0].permute(1, 2, 3, 0).cpu().numpy()
        
        # Denormalize frames if needed
        if video_frames.max() <= 1.0:
            # Denormalize using ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            video_frames = video_frames * std + mean
            video_frames = np.clip(video_frames, 0, 1)
        
        # Extract attention scores
        # Average across attention heads if multi-head
        attn_scores = attn_weights.mean(dim=1)[0].cpu().numpy()
        
        # Number of frames to visualize
        num_frames = min(10, len(video_frames))  # Show at most 10 frames
        
        # Create figure
        fig, axes = plt.subplots(2, num_frames, figsize=(16, 6))
        
        # Plot frames
        for i in range(num_frames):
            # Plot the frame
            axes[0, i].imshow(video_frames[i])
            axes[0, i].set_title(f"Frame {i+1}")
            axes[0, i].axis('off')
            
            # Plot attention heatmap
            im = axes[1, i].imshow([[attn_scores[i]]], cmap='hot', vmin=0, vmax=attn_scores.max())
            axes[1, i].set_title(f"Attention: {attn_scores[i]:.2f}")
            axes[1, i].axis('off')
        
        # Add a colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        # Add title
        class_map_rev = {0: 'Normal', 1: 'Near Collision', 2: 'Collision'}
        fig.suptitle(f"Attention Visualization for {class_map_rev[target_idx]} Sample", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(os.path.join(self.save_dir, 'attention_visualization.png'))
        plt.close()
        
        self.logger.info("Attention visualization saved")
    
    def cleanup_distributed(self):
        """Clean up distributed training resources"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


# Utility functions for distributed training setup

def setup_for_distributed_training():
    """
    Setup function for distributed training when called from command line.
    This should be called at the beginning of your training script.
    """
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"Setting up distributed training: World size={world_size}, Local rank={local_rank}")
        
        # Set CUDA device
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank,
                timeout=timedelta(hours=2)
            )
        
        return True, local_rank, world_size
    else:
        return False, 0, 1


def is_main_process():
    """Check if this is the main process (for logging and saving)"""
    return not dist.is_initialized() or dist.get_rank() == 0


# Example usage for distributed training script
def create_distributed_trainer(train_dataset, val_dataset, test_dataset, **kwargs):
    """
    Factory function to create a VideoClassifier with proper distributed setup
    
    Usage in your training script:
    ```python
    # At the beginning of your script
    distributed, local_rank, world_size = setup_for_distributed_training()
    
    # Create the classifier
    classifier = create_distributed_trainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        base_model='resnet18',
        # ... other parameters
    )
    
    # Train the model
    history = classifier.train(epochs=30)
    
    # Test the model
    test_results = classifier.test()
    
    # Clean up
    classifier.cleanup_distributed()
    ```
    """
    distributed, local_rank, world_size = setup_for_distributed_training()
    
    # Create classifier with distributed settings
    classifier = VideoClassifier(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        distributed=distributed,
        local_rank=local_rank,
        world_size=world_size,
        **kwargs
    )
    
    return classifier