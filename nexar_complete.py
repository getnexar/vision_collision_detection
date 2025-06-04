#!/usr/bin/env python3
"""
Complete Nexar Video Classification Training Script
Includes both SimpleVideoClassifier and main training script in one file.

Usage:
    # Single GPU:
    python nexar_complete.py

    # Multiple GPUs:
    torchrun --nproc_per_node=4 nexar_complete.py

    # With custom parameters:
    python nexar_complete.py --base-model resnet18 --temporal-mode attention --epochs 30

    # Grid search:
    python nexar_complete.py --run-grid-search
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import numpy as np
import random
from datetime import datetime

# Import your existing modules
try:
    from nexar_train import *
    from nexar_videos import *
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure nexar_train.py and nexar_videos.py are in the same directory")
    sys.exit(1)


# ========================================================================================
# SIMPLE VIDEO CLASSIFIER CLASS
# ========================================================================================

class SimpleVideoClassifier:
    """
    Minimal and easy to debug distributed video classifier
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
        save_dir='checkpoints'
    ):
        # Setup distributed training
        self.setup_distributed()
        
        # Basic parameters
        self.base_model = base_model
        self.temporal_mode = temporal_mode
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        
        # Setup logger
        self.setup_logger()
        
        # Create data loaders
        self.setup_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Create model
        self.create_model()
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training variables - track best training loss during training
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')  # Will be set after final validation
        
        if self.is_master:
            os.makedirs(save_dir, exist_ok=True)
            self.log(f"Setup complete. Device: {self.device}, World size: {self.world_size}")
    
    def setup_distributed(self):
        """Setup distributed training"""
        # Check if distributed
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.distributed = True
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ.get('RANK', 0))
            
            # Setup device
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            self.is_master = (self.rank == 0)
        else:
            self.distributed = False
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_master = True
    
    def setup_logger(self):
        """Setup simple logger"""
        if self.is_master:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [RANK:%(rank)d] %(message)s',
                datefmt='%H:%M:%S'
            )
            self.logger = logging.getLogger()
        else:
            # Silent logger for other ranks
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.CRITICAL)
    
    def log(self, message):
        """Simple log with rank"""
        if self.is_master:
            extra = {'rank': self.rank}
            self.logger.info(message, extra=extra)
    
    def setup_data_loaders(self, train_dataset, val_dataset, test_dataset):
        """Create data loaders"""
        # Only training needs DistributedSampler since validation runs only on master
        if self.distributed:
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_shuffle = False
        else:
            self.train_sampler = None
            train_shuffle = True
        
        # Training loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Validation loader - NO DistributedSampler since only master validates
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Test loader - same as validation
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        if self.is_master:
            self.log(f"Data loaders created: Train={len(self.train_loader)}, Val={len(self.val_loader)}")
    
    def create_model(self):
        """Create the model"""
        from nexar_arch import EnhancedFrameCNN
        
        # Create model
        self.model = EnhancedFrameCNN(
            base_model=self.base_model,
            pretrained=True,
            dropout_rate=0.5,
            temporal_mode=self.temporal_mode
        )
        
        # Adapt to number of classes
        feature_dim = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(feature_dim, self.num_classes)
        
        # Move to device
        self.model.to(self.device)
        
        # DDP wrapping
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        if self.is_master:
            self.log(f"Model created: {self.base_model} + {self.temporal_mode}, Classes: {self.num_classes}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        # Update sampler
        if self.distributed and self.train_sampler:
            self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_samples = 0
        
        # Progress bar only for master
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}") if self.is_master else self.train_loader
        
        for batch_idx, batch in enumerate(iterator):
            # Get data
            frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
            targets = batch['target']
            
            # Convert targets to numbers
            if isinstance(targets[0], str):
                class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
            else:
                targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            if self.is_master:
                avg_loss = total_loss / total_samples
                iterator.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # Clean memory
            del frames, targets, outputs, loss
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Synchronize loss between processes
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return avg_loss
    
    def validate(self):
        """Validation - only master process, others return immediately"""
        if not self.is_master:
            # Non-master processes return dummy values immediately
            return 0.0, 0.0
        
        self.log("Starting validation...")
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
                    targets = batch['target']
                    
                    if isinstance(targets[0], str):
                        class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                        targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                    else:
                        targets = targets.to(self.device)
                    
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, targets)
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == targets).sum().item()
                    total_loss += loss.item() * targets.size(0)
                    total_samples += targets.size(0)
                    
                except Exception as e:
                    self.log(f"Warning: Validation batch failed: {e}")
                    continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        self.log(f"Validation complete: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, train_loss, val_loss=None, val_acc=None, is_best=False):
        """Save checkpoint - unified method"""
        if not self.is_master:
            return
        
        # Get model without DDP wrapper
        model_to_save = self.model.module if self.distributed else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'base_model': self.base_model,
            'temporal_mode': self.temporal_mode,
            'num_classes': self.num_classes,
        }
        
        # Add validation metrics if provided
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        if val_acc is not None:
            checkpoint['val_accuracy'] = val_acc
        
        # Always save last checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'last_model.pth'))
        
        # Save best model based on training loss (during training) or validation loss (final)
        if val_loss is not None:
            # Final validation - use validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
                self.log(f"*** New best model saved! Val loss: {val_loss:.4f} ***")
        else:
            # During training - use training loss
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
                self.log(f"*** New best model saved! Train loss: {train_loss:.4f} ***")
        
        # Save epoch-specific checkpoint (except for the last epoch)
        if epoch < self.total_epochs - 1:  # Not the last epoch
            epoch_filename = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, os.path.join(self.save_dir, epoch_filename))
            self.log(f"Saved epoch {epoch+1} checkpoint: {epoch_filename}")
        
        self.log(f"Saved checkpoint for epoch {epoch+1}")
    
    def train(self, epochs=10):
        """Main training loop"""
        self.total_epochs = epochs  # Store for checkpoint saving logic
        self.log(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training - all processes participate
            train_loss = self.train_epoch(epoch)
            
            # Synchronization
            if self.distributed:
                dist.barrier()
            
            # Log results and save checkpoint (only master)
            if self.is_master:
                epoch_time = time.time() - epoch_start
                self.log(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, "
                        f"Time={epoch_time:.1f}s")
                
                # Save checkpoint for this epoch
                self.save_checkpoint(epoch, train_loss)
        
        # Final validation AFTER all training
        if self.is_master:
            self.log("Training completed! Running final validation...")
            val_loss, val_acc = self.validate()
            self.log(f"Final validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")
            
            # Save final model with validation results
            self.save_final_model(val_loss, val_acc)
    
    def save_final_model(self, val_loss, val_acc):
        """Save final model with validation results"""
        if not self.is_master:
            return
        
        # Get model without DDP wrapper
        model_to_save = self.model.module if self.distributed else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'best_train_loss': self.best_train_loss,
            'base_model': self.base_model,
            'temporal_mode': self.temporal_mode,
            'num_classes': self.num_classes,
        }
        
        # Update best validation loss and save if this is the best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            self.log(f"*** Final model is the best! Val loss: {val_loss:.4f} ***")
        
        # Always save final model
        torch.save(checkpoint, os.path.join(self.save_dir, 'final_model.pth'))
        self.log(f"Saved final model with validation loss: {val_loss:.4f}")
    
    def test(self):
        """Test on test set"""
        if not self.is_master:
            return None
        
        self.log("Starting testing...")
        
        # Load best model for testing
        model_paths = [
            os.path.join(self.save_dir, 'best_model.pth'),
            os.path.join(self.save_dir, 'final_model.pth'),
            os.path.join(self.save_dir, 'last_model.pth')
        ]
        
        checkpoint = None
        for path in model_paths:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                self.log(f"Loaded model from: {os.path.basename(path)}")
                break
        
        if checkpoint is None:
            self.log("No saved model found, using current model")
        else:
            model_to_load = self.model.module if self.distributed else self.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                try:
                    frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
                    targets = batch['target']
                    
                    if isinstance(targets[0], str):
                        class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                        targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                    else:
                        targets = targets.to(self.device)
                    
                    outputs = self.model(frames)
                    preds = torch.argmax(outputs, dim=1)
                    
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    
                except Exception as e:
                    self.log(f"Warning: Test batch failed: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        self.log(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy
    
    def cleanup(self):
        """Clean up resources"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


# ========================================================================================
# MAIN TRAINING SCRIPT
# ========================================================================================

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Nexar Video Classification - Complete Training Script')
    
    # Data parameters
    parser.add_argument('--base-dirs', nargs='+', 
                       default=["../data/research-nvidia-data/nvidia-1", "../data/research-nvidia-data/nvidia-2"],
                       help='Base directories containing the data')
    parser.add_argument('--metadata-csv', type=str, default="nvidia_delivery_to_train.csv",
                       help='Metadata CSV file path')
    parser.add_argument('--sensor-subdir', type=str, default='signals',
                       help='Sensor subdirectory name')
    parser.add_argument('--sample-strategy', type=str, default='center',
                       choices=['center', 'random', 'uniform'],
                       help='Sampling strategy for video frames')
    
    # Model parameters
    parser.add_argument('--base-model', type=str, default='convnext_tiny',
                       choices=['resnet18', 'resnet50', 'mobilenet_v3_small', 'convnext_tiny'],
                       help='Base model architecture')
    parser.add_argument('--temporal-mode', type=str, default='gru',
                       choices=['attention', 'lstm', 'gru', 'convolution', 'pooling'],
                       help='Temporal aggregation mode')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save-dir', type=str, default='model_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Grid search
    parser.add_argument('--run-grid-search', action='store_true',
                       help='Run grid search instead of single model training')
    
    return parser.parse_args()


def create_experiment_name(base_model, temporal_mode, timestamp=None):
    """Create experiment name"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_model}_{temporal_mode}_{timestamp}"


def is_main_process():
    """Check if this is the main process for logging"""
    return not ('WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1) or \
           int(os.environ.get('RANK', 0)) == 0


def log_info(message):
    """Log info only on main process"""
    if is_main_process():
        print(message)


def run_single_experiment(args):
    """Run a single training experiment"""
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = create_experiment_name(args.base_model, args.temporal_mode)
    
    log_info("=" * 80)
    log_info("NEXAR VIDEO CLASSIFICATION - COMPLETE TRAINING SCRIPT")
    log_info("=" * 80)
    log_info(f"Experiment: {args.experiment_name}")
    log_info(f"Base Model: {args.base_model}")
    log_info(f"Temporal Mode: {args.temporal_mode}")
    log_info(f"Epochs: {args.epochs}")
    log_info(f"Batch Size: {args.batch_size}")
    log_info(f"Learning Rate: {args.learning_rate}")
    
    # Check if distributed
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        log_info(f"Distributed training with {world_size} GPUs")
        log_info(f"Effective batch size: {args.batch_size * world_size}")
    else:
        log_info("Single GPU training")
    
    log_info("=" * 80)
    
    # Create save directory
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets
    log_info("Creating datasets...")
    
    train_data, val_data, test_data = create_datasets_with_manual_split(
        base_dirs=args.base_dirs,
        metadata_csv=args.metadata_csv,
        seed=args.seed,
        sensor_subdir=args.sensor_subdir,
        sample_strategy=args.sample_strategy,
        show_stats=is_main_process()
    )
    
    log_info(f"Dataset sizes:")
    log_info(f"  Train: {len(train_data)}")
    log_info(f"  Validation: {len(val_data)}")
    log_info(f"  Test: {len(test_data)}")
    
    # Create the SimpleVideoClassifier
    log_info("Creating SimpleVideoClassifier...")
    
    classifier = SimpleVideoClassifier(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        base_model=args.base_model,
        temporal_mode=args.temporal_mode,
        num_classes=3,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=os.path.join(args.save_dir, args.experiment_name)
    )
    
    log_info("Starting training...")
    
    # Train the model
    classifier.train(epochs=args.epochs)
    
    # Test the model
    log_info("Starting testing...")
    test_accuracy = classifier.test()
    
    # Print final results
    log_info("=" * 80)
    log_info("EXPERIMENT COMPLETED SUCCESSFULLY")
    log_info("=" * 80)
    log_info(f"Best training loss: {classifier.best_train_loss:.6f}")
    log_info(f"Best validation loss: {classifier.best_val_loss:.6f}")
    if test_accuracy is not None:
        log_info(f"Final test accuracy: {test_accuracy:.4f}")
    log_info(f"Results saved in: {args.save_dir}/{args.experiment_name}")
    
    # List saved checkpoints
    if is_main_process():
        checkpoint_dir = os.path.join(args.save_dir, args.experiment_name)
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            log_info(f"Saved checkpoints: {', '.join(sorted(checkpoints))}")
    
    log_info("=" * 80)
    
    # Clean up
    classifier.cleanup()
    
    return classifier, test_accuracy


def run_grid_search(args):
    """Run grid search experiments"""
    log_info("=" * 80)
    log_info("RUNNING GRID SEARCH")
    log_info("=" * 80)
    
    # Define grid search parameters
    base_models = ['resnet18', 'convnext_tiny']
    temporal_modes = ['attention', 'gru', 'lstm']
    learning_rates = [1e-4, 5e-5]
    
    results = []
    
    for base_model in base_models:
        for temporal_mode in temporal_modes:
            for lr in learning_rates:
                log_info(f"\nRunning experiment: {base_model} + {temporal_mode} + lr={lr}")
                
                # Update args for this experiment
                args.base_model = base_model
                args.temporal_mode = temporal_mode
                args.learning_rate = lr
                args.experiment_name = create_experiment_name(
                    base_model, 
                    temporal_mode, 
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{str(lr).replace('.', '_')}"
                )
                
                try:
                    classifier, test_accuracy = run_single_experiment(args)
                    
                    if is_main_process():
                        results.append({
                            'base_model': base_model,
                            'temporal_mode': temporal_mode,
                            'learning_rate': lr,
                            'experiment_name': args.experiment_name,
                            'test_accuracy': test_accuracy,
                            'best_train_loss': classifier.best_train_loss,
                            'best_val_loss': classifier.best_val_loss
                        })
                        
                except Exception as e:
                    log_info(f"Error in experiment {base_model}+{temporal_mode}+{lr}: {e}")
                    continue
    
    # Print grid search summary
    if is_main_process():
        log_info("=" * 80)
        log_info("GRID SEARCH RESULTS SUMMARY")
        log_info("=" * 80)
        
        # Sort by test accuracy (descending)
        results_sorted = sorted(results, key=lambda x: x['test_accuracy'] or 0, reverse=True)
        
        for i, result in enumerate(results_sorted, 1):
            log_info(f"{i}. {result['base_model']} + {result['temporal_mode']} + lr={result['learning_rate']}")
            log_info(f"   Best Train Loss: {result['best_train_loss']:.6f}")
            log_info(f"   Best Val Loss: {result['best_val_loss']:.6f}")
            log_info(f"   Test Accuracy: {result['test_accuracy']:.4f}")
            log_info(f"   Experiment: {result['experiment_name']}")
            log_info("")
        
        # Find best result
        if results_sorted:
            best = results_sorted[0]
            log_info("BEST RESULT:")
            log_info(f"Model: {best['base_model']} + {best['temporal_mode']}")
            log_info(f"Learning Rate: {best['learning_rate']}")
            log_info(f"Test Accuracy: {best['test_accuracy']:.4f}")
            log_info(f"Validation Loss: {best['best_val_loss']:.6f}")
    
    return results


def run_notebook_equivalent():
    """
    Function that replicates your notebook workflow
    Call this from a notebook or interactive Python session
    """
    # Your original notebook parameters
    seed = 42
    set_random_seeds(seed)
    
    # Parameters
    base_dirs = ["../data/research-nvidia-data/nvidia-1", "../data/research-nvidia-data/nvidia-2"]
    metadata_csv = "nvidia_delivery_to_train.csv"
    base_model = "convnext_tiny"
    temporal_mode = "gru"
    batch_size = 8
    learning_rate = 1e-4
    epochs = 15
    experiment_name = f"{base_model}_{temporal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = "model_results"
    
    print("Creating datasets...")
    
    # Create datasets
    train_data, val_data, test_data = create_datasets_with_manual_split(
        base_dirs=base_dirs,
        metadata_csv=metadata_csv,
        seed=seed,
        sensor_subdir='signals',
        sample_strategy='center',
        show_stats=True
    )
    
    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create classifier
    print("Creating SimpleVideoClassifier...")
    
    classifier = SimpleVideoClassifier(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        base_model=base_model,
        temporal_mode=temporal_mode,
        num_classes=3,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir=os.path.join(save_dir, experiment_name)
    )
    
    print("Starting training...")
    
    # Train and test
    classifier.train(epochs=epochs)
    test_accuracy = classifier.test()
    
    print(f"Training completed! Test accuracy: {test_accuracy:.4f}")
    print(f"Best training loss: {classifier.best_train_loss:.6f}")
    print(f"Best validation loss: {classifier.best_val_loss:.6f}")
    print(f"Results saved in: {save_dir}/{experiment_name}")
    
    return classifier, test_accuracy


def test_single_gpu():
    """Quick test function for single GPU debugging"""
    print("Running quick test...")
    
    # Minimal test arguments
    class TestArgs:
        base_dirs = ["../data/research-nvidia-data/nvidia-1"]
        metadata_csv = "nvidia_delivery_to_train.csv"
        sensor_subdir = 'signals'
        sample_strategy = 'center'
        base_model = 'resnet18'
        temporal_mode = 'attention'
        epochs = 2  # Just 2 epochs for quick test
        batch_size = 4
        learning_rate = 1e-4
        experiment_name = f"test_{datetime.now().strftime('%H%M%S')}"
        save_dir = 'test_results'
        seed = 42
        run_grid_search = False
    
    args = TestArgs()
    
    try:
        classifier, test_accuracy = run_single_experiment(args)
        print(f"Test completed successfully! Accuracy: {test_accuracy:.4f}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    args = parse_args()
    
    try:
        if args.run_grid_search:
            results = run_grid_search(args)
        else:
            classifier, test_accuracy = run_single_experiment(args)
            
    except Exception as e:
        log_info(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()