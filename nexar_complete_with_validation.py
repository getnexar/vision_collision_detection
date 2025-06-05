#!/usr/bin/env python3
"""
Complete Nexar Video Classification Training Script with Fixed Distributed Validation
Includes validation after each epoch without deadlock issues.

Usage:
    # Single GPU:
    python nexar_complete_with_validation.py

    # Multiple GPUs:
    torchrun --nproc_per_node=4 nexar_complete_with_validation.py

    # With custom parameters:
    python nexar_complete_with_validation.py --base-model resnet18 --temporal-mode attention --epochs 30

    # Grid search:
    python nexar_complete_with_validation.py --run-grid-search
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
from sklearn.metrics import precision_recall_fscore_support, classification_report
# Import your existing modules
try:
    from nexar_train import *
    from nexar_videos import *
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure nexar_train.py and nexar_videos.py are in the same directory")
    sys.exit(1)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ========================================================================================
# SIMPLE VIDEO CLASSIFIER CLASS WITH FIXED VALIDATION
# ========================================================================================

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
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, classification_report


class VideoClassifier:
    """
    Video classifier with comprehensive validation metrics and history tracking
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
       self.class_names = ['Normal', 'Near Collision', 'Collision']
       
       # Setup logger
       self.setup_logger()
       
       # Create data loaders
       self.setup_data_loaders(train_dataset, val_dataset, test_dataset)
       
       # Create model
       self.create_model()
       
       # Setup loss and optimizer
       self.criterion = nn.CrossEntropyLoss()
       self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
       
       # Training variables
       self.best_val_loss = float('inf')
       self.training_history = {
           'epoch': [],
           'train_loss': [],
           'val_loss': [],
           'val_accuracy': [],
           'val_precision_normal': [],
           'val_precision_near_collision': [],
           'val_precision_collision': [],
           'val_recall_normal': [],
           'val_recall_near_collision': [],
           'val_recall_collision': [],
           'val_f1_normal': [],
           'val_f1_near_collision': [],
           'val_f1_collision': [],
           'epoch_time': []
       }
       
       if self.is_master:
           os.makedirs(save_dir, exist_ok=True)
           self.log(f"Setup complete. Device: {self.device}, World size: {self.world_size}")
    
    def setup_distributed(self):
       """Setup distributed training"""
       if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
           self.distributed = True
           self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
           self.world_size = int(os.environ['WORLD_SIZE'])
           self.rank = int(os.environ.get('RANK', 0))
           
           self.device = torch.device(f"cuda:{self.local_rank}")
           torch.cuda.set_device(self.device)
            
           if not dist.is_initialized():
                from datetime import timedelta
                dist.init_process_group(backend='nccl', timeout=timedelta(hours=12))

           
           self.is_master = (self.rank == 0)
       else:
           self.distributed = False
           self.local_rank = 0
           self.world_size = 1
           self.rank = 0
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.is_master = True
    
    def setup_logger(self):
       """Setup logger"""
       if self.is_master:
           logging.basicConfig(
               level=logging.INFO,
               format='%(asctime)s [RANK:%(rank)d] %(message)s',
               datefmt='%H:%M:%S'
           )
           self.logger = logging.getLogger()
       else:
           self.logger = logging.getLogger()
           self.logger.setLevel(logging.CRITICAL)
    
    def log(self, message):
       """Log with rank"""
       if self.is_master:
           extra = {'rank': self.rank}
           self.logger.info(message, extra=extra)
    
    def setup_data_loaders(self, train_dataset, val_dataset, test_dataset):
       """Create data loaders"""
       if self.distributed:
           self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
           train_shuffle = False
       else:
           self.train_sampler = None
           train_shuffle = True
       
       self.train_loader = DataLoader(
           train_dataset,
           batch_size=self.batch_size,
           shuffle=train_shuffle,
           sampler=self.train_sampler,
           num_workers=0,
           pin_memory=True,
           drop_last=True
       )
       
       # Validation loader - only master validates
       self.val_loader = DataLoader(
           val_dataset,
           batch_size=self.batch_size,
           shuffle=False,
           num_workers=0,
           pin_memory=True
       )
       
       self.test_loader = DataLoader(
           test_dataset,
           batch_size=self.batch_size,
           shuffle=False,
           num_workers=0
       )
       
       if self.is_master:
           self.log(f"Data loaders created: Train={len(self.train_loader)}, Val={len(self.val_loader)}")
       
    def create_model(self):
        """Create the model"""
        from nexar_arch import EnhancedFrameCNN
    
        self.model = EnhancedFrameCNN(
            base_model=self.base_model,
            pretrained=True,
            dropout_rate=0.5,
            temporal_mode=self.temporal_mode
        )
    
        # Adapt to number of classes
        feature_dim = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(feature_dim, self.num_classes)
    
        self.model.to(self.device)
    
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
        if self.is_master:
            self.log(f"Model created: {self.base_model} + {self.temporal_mode}, Classes: {self.num_classes}")
    
    def train(self, epochs=10):
        """Main training loop with comprehensive validation"""
        self.log(f"Starting training for {epochs} epochs")
    
        for epoch in range(epochs):
            epoch_start_time = time.time()
    
            # Training
            train_loss = self.train_epoch(epoch)
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            # Validation (only master)
            val_metrics = {}
            if self.is_master:
                val_metrics = self.validate()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            # Synchronization barrier
            if self.distributed:
                dist.barrier()
    
            epoch_time = time.time() - epoch_start_time
    
            # Update history (only master)
            if self.is_master:
                self.training_history['epoch'].append(epoch + 1)
                self.training_history['train_loss'].append(train_loss)
                self.training_history['epoch_time'].append(epoch_time)
    
                for key, value in val_metrics.items():
                    self.training_history[key].append(value)
    
                # Log epoch summary
                self.log(f"Epoch {epoch+1}/{epochs} Complete:")
                self.log(f"  Train Loss: {train_loss:.4f}")
                self.log(f"  Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
                self.log(f"  Val Accuracy: {val_metrics.get('val_accuracy', 'N/A'):.4f}")
                self.log(f"  Epoch Time: {epoch_time:.1f}s")
    
                # Save checkpoint and history
                self.save_checkpoint(epoch, train_loss, val_metrics)
                self.save_training_history_csv()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        if self.is_master:
            self.log("Training completed!")
            self.log(f"Best validation loss achieved: {self.best_val_loss:.4f}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
    
        if self.distributed and self.train_sampler:
            self.train_sampler.set_epoch(epoch)
    
        total_loss = 0.0
        total_samples = 0
    
        iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}") if self.is_master else self.train_loader
    
        for batch in iterator:
            frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
            targets = batch['target']
    
            if isinstance(targets[0], str):
                class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
            else:
                targets = targets.to(self.device)
    
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
            if self.is_master:
                avg_loss = total_loss / total_samples
                iterator.set_postfix({'loss': f"{avg_loss:.4f}"})
    
            del frames, targets, outputs, loss
    
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
        # Synchronize loss between processes
        if self.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
    
        return avg_loss
    
    def validate(self):
        """Comprehensive validation with per-class metrics"""
        if not self.is_master:
            return {}
        self.log(f"Starting validation on {len(self.val_loader)} batches...")
    
        # Use non-DDP model for validation
        model_for_val = self.model.module if hasattr(self.model, 'module') else self.model
        model_for_val.eval()
    
        total_loss = 0.0
        all_preds = []
        all_targets = []
        total_samples = 0
    
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", disable=not self.is_master):
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
                targets = batch['target']
    
                if isinstance(targets[0], str):
                    class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                else:
                    targets = targets.to(self.device)
    
                outputs = model_for_val(frames)
                loss = self.criterion(outputs, targets)
    
                preds = torch.argmax(outputs, dim=1)
    
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)
    
                del frames, targets, outputs, loss
    
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
        # Per-class precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0, labels=[0, 1, 2]
        )
    
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_precision_normal': precision[0],
            'val_precision_near_collision': precision[1],
            'val_precision_collision': precision[2],
            'val_recall_normal': recall[0],
            'val_recall_near_collision': recall[1],
            'val_recall_collision': recall[2],
            'val_f1_normal': f1[0],
            'val_f1_near_collision': f1[1],
            'val_f1_collision': f1[2]
        }
    
        # Log detailed results
        self.log(f"Validation Results:")
        self.log(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        self.log(f"  Per-class Precision: Normal={precision[0]:.4f}, Near Collision={precision[1]:.4f}, Collision={precision[2]:.4f}")
        self.log(f"  Per-class Recall: Normal={recall[0]:.4f}, Near Collision={recall[1]:.4f}, Collision={recall[2]:.4f}")
        self.log(f"  Per-class F1: Normal={f1[0]:.4f}, Near Collision={f1[1]:.4f}, Collision={f1[2]:.4f}")
    
        return metrics
    
    def save_training_history_csv(self):
       """Save training history to CSV"""
       if not self.is_master or not self.training_history['epoch']:
           return
       
       df = pd.DataFrame(self.training_history)
       csv_path = os.path.join(self.save_dir, 'training_history.csv')
       df.to_csv(csv_path, index=False)
       self.log(f"Training history saved to: {csv_path}")
    
    def save_checkpoint(self, epoch, train_loss, val_metrics=None):
       """Save checkpoint"""
       if not self.is_master:
           return
       
       model_to_save = self.model.module if self.distributed else self.model
       
       checkpoint = {
           'epoch': epoch,
           'model_state_dict': model_to_save.state_dict(),
           'optimizer_state_dict': self.optimizer.state_dict(),
           'train_loss': train_loss,
           'best_val_loss': self.best_val_loss,
           'base_model': self.base_model,
           'temporal_mode': self.temporal_mode,
           'num_classes': self.num_classes,
           'training_history': self.training_history
       }
       
       if val_metrics:
           checkpoint.update(val_metrics)
       
       # Save epoch-specific checkpoint
       epoch_filename = f'checkpoint_epoch_{epoch+1}.pth'
       torch.save(checkpoint, os.path.join(self.save_dir, epoch_filename))
       
       # Save as latest
       torch.save(checkpoint, os.path.join(self.save_dir, 'latest_model.pth'))
       
       # Save as best if this is the best validation loss
       if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
           self.best_val_loss = val_metrics['val_loss']
           torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
           self.log(f"*** NEW BEST MODEL! Validation Loss: {val_metrics['val_loss']:.4f} ***")
       
       self.log(f"Saved checkpoint: {epoch_filename}")
       
    def test(self):
       """Test on test set"""
       if not self.is_master:
           return None
       
       self.log("Starting testing...")
       
       # Load best model
       best_model_path = os.path.join(self.save_dir, 'best_model.pth')
       if os.path.exists(best_model_path):
           checkpoint = torch.load(best_model_path, map_location=self.device)
           model_to_load = self.model.module if self.distributed else self.model
           model_to_load.load_state_dict(checkpoint['model_state_dict'])
           self.log(f"Loaded best model for testing (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
       else:
           self.log("No best model found, using current model")
       
       # Use non-DDP model for testing
       model_for_test = self.model.module if hasattr(self.model, 'module') else self.model
       model_for_test.eval()
       
       all_preds = []
       all_targets = []
       
       with torch.no_grad():
           for batch in tqdm(self.test_loader, desc="Testing"):
               frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
               targets = batch['target']
               
               if isinstance(targets[0], str):
                   class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                   targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
               else:
                   targets = targets.to(self.device)
               
               outputs = model_for_test(frames)
               preds = torch.argmax(outputs, dim=1)
               
               all_preds.extend(preds.cpu().numpy())
               all_targets.extend(targets.cpu().numpy())
       
       # Calculate test metrics
       accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
       precision, recall, f1, _ = precision_recall_fscore_support(
           all_targets, all_preds, average=None, zero_division=0, labels=[0, 1, 2]
       )
       
       # Log detailed test results
       self.log(f"Test Results:")
       self.log(f"  Overall Accuracy: {accuracy:.4f}")
       self.log(f"  Per-class Results:")
       for i, class_name in enumerate(self.class_names):
           self.log(f"    {class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
       
       # Save test results
       test_results = {
           'test_accuracy': accuracy,
           'test_precision_normal': precision[0],
           'test_precision_near_collision': precision[1],
           'test_precision_collision': precision[2],
           'test_recall_normal': recall[0],
           'test_recall_near_collision': recall[1],
           'test_recall_collision': recall[2],
           'test_f1_normal': f1[0],
           'test_f1_near_collision': f1[1],
           'test_f1_collision': f1[2]
       }
       
       # Save test results to CSV
       test_df = pd.DataFrame([test_results])
       test_csv_path = os.path.join(self.save_dir, 'test_results.csv')
       test_df.to_csv(test_csv_path, index=False)
       self.log(f"Test results saved to: {test_csv_path}")
       
       return accuracy
    
    def cleanup(self):
       """Clean up resources"""
       if self.distributed and dist.is_initialized():
           dist.destroy_process_group()

# ========================================================================================
# MAIN TRAINING SCRIPT
# ========================================================================================

import random

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
   parser.add_argument('--metadata-csv', type=str, default="nvidia_new_train.csv",
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
   log_info("NEXAR VIDEO CLASSIFICATION - COMPREHENSIVE TRAINING WITH VALIDATION")
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
   
   # Create the VideoClassifier
   log_info("Creating VideoClassifier...")
   
   classifier = VideoClassifier(
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
   if test_accuracy is not None:
       log_info(f"Final test accuracy: {test_accuracy:.4f}")
   log_info(f"Best validation loss: {classifier.best_val_loss:.4f}")
   log_info(f"Results saved in: {args.save_dir}/{args.experiment_name}")
   
   # List saved files
   if is_main_process():
       checkpoint_dir = os.path.join(args.save_dir, args.experiment_name)
       if os.path.exists(checkpoint_dir):
           files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.csv'))]
           log_info(f"Saved files: {', '.join(sorted(files))}")
   
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
                       # Get final validation metrics from training history
                       final_val_acc = classifier.training_history['val_accuracy'][-1] if classifier.training_history['val_accuracy'] else 0
                       
                       results.append({
                           'base_model': base_model,
                           'temporal_mode': temporal_mode,
                           'learning_rate': lr,
                           'experiment_name': args.experiment_name,
                           'test_accuracy': test_accuracy,
                           'best_val_loss': classifier.best_val_loss,
                           'final_val_acc': final_val_acc
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
           log_info(f"   Best Val Loss: {result['best_val_loss']:.4f}")
           log_info(f"   Final Val Acc: {result['final_val_acc']:.4f}")
           log_info(f"   Test Accuracy: {result['test_accuracy']:.4f}")
           log_info(f"   Experiment: {result['experiment_name']}")
           log_info("")
       
       # Save grid search results
       if results_sorted:
           grid_df = pd.DataFrame(results_sorted)
           grid_csv_path = os.path.join(args.save_dir, f'grid_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
           grid_df.to_csv(grid_csv_path, index=False)
           log_info(f"Grid search results saved to: {grid_csv_path}")
           
           # Best result
           best = results_sorted[0]
           log_info("BEST RESULT:")
           log_info(f"Model: {best['base_model']} + {best['temporal_mode']}")
           log_info(f"Learning Rate: {best['learning_rate']}")
           log_info(f"Test Accuracy: {best['test_accuracy']:.4f}")
           log_info(f"Validation Loss: {best['best_val_loss']:.4f}")
   
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
   metadata_csv = "nvidia_new_train.csv"
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
   print("Creating VideoClassifier...")
   
   classifier = VideoClassifier(
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
   
   print("Starting training with validation after each epoch...")
   
   # Train and test
   classifier.train(epochs=epochs)
   test_accuracy = classifier.test()
   
   print(f"Training completed! Test accuracy: {test_accuracy:.4f}")
   print(f"Best validation loss: {classifier.best_val_loss:.4f}")
   print(f"Results saved in: {save_dir}/{experiment_name}")
   
   # Plot training history if in notebook
   try:
       import matplotlib.pyplot as plt
       
       # Check if we have validation data
       if classifier.training_history['val_loss']:
           fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
           
           epochs_range = classifier.training_history['epoch']
           
           # Plot losses
           ax1.plot(epochs_range, classifier.training_history['train_loss'], 'b-', label='Train Loss')
           ax1.plot(epochs_range, classifier.training_history['val_loss'], 'r-', label='Val Loss')
           ax1.set_xlabel('Epoch')
           ax1.set_ylabel('Loss')
           ax1.set_title('Training and Validation Loss')
           ax1.legend()
           ax1.grid(True)
           
           # Plot accuracy
           ax2.plot(epochs_range, classifier.training_history['val_accuracy'], 'g-', label='Val Accuracy')
           ax2.set_xlabel('Epoch')
           ax2.set_ylabel('Accuracy')
           ax2.set_title('Validation Accuracy')
           ax2.legend()
           ax2.grid(True)
           
           # Plot per-class precision
           ax3.plot(epochs_range, classifier.training_history['val_precision_normal'], 'b-', label='Normal')
           ax3.plot(epochs_range, classifier.training_history['val_precision_near_collision'], 'orange', label='Near Collision')
           ax3.plot(epochs_range, classifier.training_history['val_precision_collision'], 'r-', label='Collision')
           ax3.set_xlabel('Epoch')
           ax3.set_ylabel('Precision')
           ax3.set_title('Per-Class Precision')
           ax3.legend()
           ax3.grid(True)
           
           # Plot per-class recall
           ax4.plot(epochs_range, classifier.training_history['val_recall_normal'], 'b-', label='Normal')
           ax4.plot(epochs_range, classifier.training_history['val_recall_near_collision'], 'orange', label='Near Collision')
           ax4.plot(epochs_range, classifier.training_history['val_recall_collision'], 'r-', label='Collision')
           ax4.set_xlabel('Epoch')
           ax4.set_ylabel('Recall')
           ax4.set_title('Per-Class Recall')
           ax4.legend()
           ax4.grid(True)
           
           plt.tight_layout()
           plt.show()
       
   except ImportError:
       print("Matplotlib not available - skipping plots")
   
   return classifier, test_accuracy


def test_single_gpu():
   """Quick test function for single GPU debugging"""
   print("Running quick test...")
   
   # Minimal test arguments
   class TestArgs:
       base_dirs = ["../data/research-nvidia-data/nvidia-1"]
       metadata_csv = "nvidia_new_train.csv"
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
       print(f"Validation loss history: {classifier.training_history['val_loss']}")
       print(f"Files saved in: {args.save_dir}/{args.experiment_name}")
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