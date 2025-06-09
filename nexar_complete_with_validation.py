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

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Standard libraries
import os
import sys
import time
import argparse
import logging
import random
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import cv2
import decord
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

from nexar_video_aug import create_video_transforms

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.multiprocessing.set_sharing_strategy('file_system')

# ========================================================================================
# SIMPLE VIDEO CLASSIFIER CLASS WITH FIXED VALIDATION
# ========================================================================================

class VideoDataset(Dataset):
    """Simple optimized video dataset - no caching overhead"""

    def __init__(self, video_paths, labels, video_ids=None, fps=10, duration=5, 
                 is_train=True, transform=None, sample_strategy='metadata_center', 
                 center_time_column=None, metadata_df=None):
        """
        Args:
            video_paths: List of full paths to video files
            labels: List of labels corresponding to each video
            video_ids: Optional list of video IDs
            fps: Frames per second to extract
            duration: Duration in seconds to extract
            is_train: Whether this is training mode
            transform: Video transforms to apply
            sample_strategy: 'random', 'center', or 'metadata_center'
            center_time_column: Column name for center time (for 'metadata_center')
            metadata_df: DataFrame with metadata (required for 'metadata_center')
        """
        self.video_paths = video_paths
        self.labels = labels
        self.video_ids = video_ids if video_ids is not None else list(range(len(video_paths)))
        self.fps = fps
        self.duration = duration
        self.is_train = is_train
        self.transform = transform
        self.sample_strategy = sample_strategy
        self.center_time_column = center_time_column
        self.metadata_df = metadata_df
        
        # Pre-compute only metadata that we need for metadata_center strategy
        self._metadata_cache = {}
        if sample_strategy == 'metadata_center':
            self._precompute_fps_only()
        
        # Validate inputs
        assert len(video_paths) == len(labels), "video_paths and labels must have same length"
        assert sample_strategy in ['random', 'center', 'metadata_center'], \
            "sample_strategy must be 'random', 'center', or 'metadata_center'"
        
        if sample_strategy == 'metadata_center':
            assert metadata_df is not None, "metadata_df required for 'metadata_center' strategy"
            assert center_time_column is not None, "center_time_column required for 'metadata_center' strategy"
            assert center_time_column in metadata_df.columns, f"Column '{center_time_column}' not found in metadata"

    def __len__(self):
        return len(self.video_paths)

    def _precompute_fps_only(self):
        """Pre-compute only FPS for metadata_center strategy"""
        print("Pre-computing FPS for metadata_center strategy...")
        for video_path in self.video_paths:
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                self._metadata_cache[video_path] = fps if fps > 0 else 30.0
            except Exception:
                self._metadata_cache[video_path] = 30.0

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_id = self.video_ids[idx]
        
        try:
            # Open VideoReader - simple and direct
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            num_frames = len(vr)
            frames_needed = self.fps * self.duration
            
            # Determine start frame based on strategy
            if self.sample_strategy == 'metadata_center':
                center_time = self._get_center_time_from_metadata(idx)
                if center_time is not None:
                    # Use pre-computed FPS
                    video_fps = self._metadata_cache.get(video_path, 30.0)
                    center_frame = int(center_time * video_fps)
                    frames_half = frames_needed // 2
                    start_frame = max(0, center_frame - frames_half)
                    if start_frame + frames_needed > num_frames:
                        start_frame = max(0, num_frames - frames_needed)
                else:
                    start_frame = self._get_random_start_frame(num_frames, frames_needed)
                    
            elif self.sample_strategy == 'center':
                if num_frames > frames_needed:
                    center_frame = num_frames // 2
                    frames_half = frames_needed // 2
                    start_frame = max(0, center_frame - frames_half)
                    if start_frame + frames_needed > num_frames:
                        start_frame = max(0, num_frames - frames_needed)
                else:
                    start_frame = 0
            else:  # random
                start_frame = self._get_random_start_frame(num_frames, frames_needed)
            
            # Ensure start_frame is within bounds
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = min(start_frame + frames_needed, num_frames)
            
            # Extract frames
            indices = list(range(start_frame, end_frame))
            frames = vr.get_batch(indices)
            
            # Convert to numpy
            if hasattr(frames, 'asnumpy'):
                frames = frames.asnumpy()
            elif isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            
            # Ensure correct frame count
            frames = self._ensure_frame_count(frames, frames_needed)
            
            # Convert to torch tensor [C, T, H, W]
            frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
            
            # Apply transforms
            if self.transform:
                frames = self.transform(frames)
            else:
                frames = frames.float() / 255.0
            
            # Convert back to [T, H, W, C]
            frames = frames.permute(1, 2, 3, 0)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            channels = 3
            if self.transform:
                final_size = 224
                frames = torch.zeros(self.fps * self.duration, final_size, final_size, channels)
            else:
                frames = torch.zeros(self.fps * self.duration, 720, 1280, channels)
        
        return {
            'frames': frames,
            'target': label,
            'id': video_id
        }

    def _get_random_start_frame(self, num_frames, frames_needed):
        """Get random start frame"""
        if num_frames > frames_needed:
            return random.randint(0, num_frames - frames_needed)
        return 0
    
    def _get_center_time_from_metadata(self, idx):
        """Get center time from metadata for the given index"""
        if self.metadata_df is None or self.center_time_column is None:
            return None
            
        try:
            video_id = self.video_ids[idx]
            metadata_row = self.metadata_df[self.metadata_df['id'] == video_id]
            if len(metadata_row) > 0:
                center_time = metadata_row.iloc[0][self.center_time_column]
                return center_time if not pd.isna(center_time) else None
        except Exception:
            pass
        return None
    
    def _ensure_frame_count(self, frames, target_count):
        """Ensure frames array has exactly target_count frames"""
        current_count = len(frames)
        
        if current_count < target_count:
            if current_count > 0:
                last_frame = frames[-1]
                padding = np.repeat(last_frame[np.newaxis, :], target_count - current_count, axis=0)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                h, w = 720, 1280
                frames = np.zeros((target_count, h, w, 3), dtype=np.uint8)
        elif current_count > target_count:
            frames = frames[:target_count]
            
        return frames

    def show_batch(self, batch=None, m=4, rows_per_page=2, fps=10, normalize=True, 
                   temp_dir="./temp_videos", video_width=240, **kwargs):
        """
        Display a batch of videos in a grid layout
        
        Args:
            batch: Batch of data to display (if None, will sample from dataset)
            m: Number of videos per row in the grid
            rows_per_page: Number of rows to display
            fps: Frames per second for the output videos
            normalize: Whether to denormalize the frames for display
            temp_dir: Directory to store temporary video files
            video_width: Width of videos in the HTML display
            
        Returns:
            HTML display object
        """
        import os
        import torch
        import numpy as np
        import uuid
        from pathlib import Path
        from IPython.display import HTML, display
        from torch.utils.data import DataLoader
        import imageio
        
        # Create directories
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        # Create a unique subfolder to avoid conflicts
        session_id = str(uuid.uuid4())[:8]
        session_dir = temp_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Get batch if not provided
        total_videos = m * rows_per_page
        if batch is None:
            temp_loader = DataLoader(self, batch_size=total_videos, shuffle=True, num_workers=8)
            batch = next(iter(temp_loader))
        
        frames = batch['frames']
        targets = batch['target']
        ids = batch['id']
        
        # Limit to requested number of videos
        n_videos = min(len(frames), total_videos)
        
        # Convert tensors to numpy arrays if needed
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        
        # Get unique classes and assign colors
        unique_classes = list(set([str(t) for t in targets]))
        color_palette = ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        class_colors = {cls: color_palette[i % len(color_palette)] 
                       for i, cls in enumerate(unique_classes)}
        
        # Process videos
        video_paths = []
        video_titles = []
        
        # Define mean and std for denormalization
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        
        for i in range(n_videos):
            # Get video frames
            video_frames = frames[i].copy()
            
            # Denormalization
            if normalize and (video_frames.min() < 0 or video_frames.max() > 1.0):
                mean_reshaped = mean.reshape(1, 1, 1, 3)
                std_reshaped = std.reshape(1, 1, 1, 3)
                video_frames = video_frames * std_reshaped + mean_reshaped
                video_frames = np.clip(video_frames, 0, 1)
                video_frames = (video_frames * 255).astype(np.uint8)
            elif normalize and video_frames.max() <= 1.0:
                video_frames = (video_frames * 255).astype(np.uint8)
            else:
                video_frames = video_frames.astype(np.uint8)
            
            # Ensure correct shape
            if len(video_frames.shape) == 3:
                video_frames = np.repeat(video_frames[..., np.newaxis], 3, axis=-1)
            elif video_frames.shape[-1] == 1:
                video_frames = np.repeat(video_frames, 3, axis=-1)
            
            # Create title
            target_value = str(targets[i])
            title = f"{target_value} (ID: {ids[i]})"
            video_titles.append(title)
            
            # Ensure even dimensions for video encoding
            processed_frames = []
            for frame in video_frames:
                height, width = frame.shape[:2]
                if height % 2 != 0 or width % 2 != 0:
                    new_height = height + (height % 2)
                    new_width = width + (width % 2)
                    padded_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    padded_frame[:height, :width] = frame
                    processed_frames.append(padded_frame)
                else:
                    processed_frames.append(frame)
            
            # Save video
            temp_video_path = str(session_dir / f"video_{i}.mp4")
            video_paths.append(temp_video_path)
            
            try:
                imageio.mimwrite(temp_video_path, processed_frames, fps=fps, macro_block_size=1)
            except Exception as e:
                print(f"Error creating video {i}: {e}")
        
        # Create HTML with proper grid layout
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
.video-grid {{
    display: grid;
    grid-template-columns: repeat({m}, 1fr);
    gap: 15px;
    margin: 20px auto;
    max-width: 1200px;
}}
.video-item {{
    text-align: center;
}}
.video-title {{
    font-weight: bold;
    font-size: 14px;
    margin-bottom: 8px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
video {{
    width: {video_width}px;
    border: 2px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}}
</style>
</head>
<body>
<h2>Video Batch</h2>
<div class="video-grid">
"""
        
        # Add videos to grid
        for i, (path, title) in enumerate(zip(video_paths, video_titles)):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                target_value = str(targets[i])
                color = class_colors.get(target_value, '#000000')
                
                # Encode video as base64
                try:
                    from base64 import b64encode
                    with open(path, 'rb') as f:
                        mp4_data = f.read()
                    data_url = f"data:video/mp4;base64,{b64encode(mp4_data).decode()}"
                    
                    html_content += f"""
<div class="video-item">
    <div class="video-title" style="color: {color};">{title}</div>
    <video controls autoplay loop muted playsinline>
        <source src="{data_url}" type="video/mp4">
    </video>
</div>
"""
                except Exception as e:
                    print(f"Error encoding video {i}: {e}")
        
        html_content += """
</div>
</body>
</html>
"""
        
        # Display HTML
        display_html = HTML(html_content)
        display(display_html)
        
        # Clean up temporary files
        import shutil
        try:
            import time
            time.sleep(1)
            shutil.rmtree(session_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")
        
        return display_html


def create_datasets_with_manual_split(metadata_df, video_path_column='video_path', 
                                    label_column='video_type', id_column='id',
                                    split_column='split', transform_train=None, 
                                    transform_val=None, sample_strategy='random',
                                    center_time_column=None, **dataset_kwargs):
    """
    Create train/val/test datasets from metadata DataFrame with manual split
    
    Args:
        metadata_df: DataFrame containing video paths, labels, IDs, and split information
        video_path_column: Column name containing video file paths
        label_column: Column name containing labels
        id_column: Column name containing video IDs
        split_column: Column name containing split info ('train', 'val', 'test')
        transform_train: Transform for training data
        transform_val: Transform for validation/test data
        sample_strategy: Sampling strategy ('random', 'center', 'metadata_center')
        center_time_column: Column for center time (if using 'metadata_center')
        **dataset_kwargs: Additional arguments for VideoDataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Validate inputs
    required_columns = [video_path_column, label_column, id_column, split_column]
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Split data based on metadata
    train_df = metadata_df[metadata_df[split_column].str.lower() == 'train']
    val_df = metadata_df[metadata_df[split_column].str.lower() == 'val']
    test_df = metadata_df[metadata_df[split_column].str.lower() == 'test']
    
    # Create datasets
    def create_split_dataset(df, is_train=False):
        if len(df) == 0:
            return None
        return VideoDataset(
            video_paths=df[video_path_column].tolist(),
            labels=df[label_column].tolist(),
            video_ids=df[id_column].tolist(),
            is_train=is_train,
            transform=transform_train if is_train else transform_val,
            sample_strategy=sample_strategy,
            center_time_column=center_time_column,
            metadata_df=metadata_df,
            **dataset_kwargs
        )
    
    train_dataset = create_split_dataset(train_df, is_train=True)
    val_dataset = create_split_dataset(val_df, is_train=False)
    test_dataset = create_split_dataset(test_df, is_train=False)
    
    return train_dataset, val_dataset, test_dataset

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
        """
        Replace existing function - now with DistributedSampler for validation too
        """
        # Training data loader (same as before)
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
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True
        )
        
        # Validation data loader - now distributed too!
        if self.distributed:
            self.val_sampler = DistributedSampler(
                val_dataset, 
                shuffle=False,  # Important! Validation must be deterministic
                drop_last=False  # We want to see all samples
            )
            val_sampler = self.val_sampler
        else:
            val_sampler = None
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )
        
        # Test data loader (unchanged)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
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
        """Main training loop with distributed validation"""
        self.log(f"Starting training for {epochs} epochs with DISTRIBUTED validation")
    
        for epoch in range(epochs):
            epoch_start_time = time.time()
    
            # Training
            train_loss = self.train_epoch(epoch)
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            # Distributed Validation - ALL GPUs participate!
            val_metrics = self.validate()
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            # Synchronization barrier
            if self.distributed:
                dist.barrier()
    
            epoch_time = time.time() - epoch_start_time
    
            # Update history (only master saves, but all have the same data)
            if self.is_master:
                self.training_history['epoch'].append(epoch + 1)
                self.training_history['train_loss'].append(train_loss)
                self.training_history['epoch_time'].append(epoch_time)
    
                for key, value in val_metrics.items():
                    self.training_history[key].append(value)
    
                # Log epoch summary
                self.log(f"Epoch {epoch+1}/{epochs} Complete (DISTRIBUTED VALIDATION):")
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
            self.log("Training completed with DISTRIBUTED VALIDATION!")
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
        """
        Distributed validation - replace existing validate() function
        All GPUs work in parallel, results are identical to master-only approach
        """
        self.log(f"Starting distributed validation on {len(self.val_loader)} batches per GPU...")
        
        # Use non-DDP model for validation
        model_for_val = self.model.module if hasattr(self.model, 'module') else self.model
        model_for_val.eval()
        
        # Set epoch for validation sampler (for deterministic behavior)
        if self.distributed and hasattr(self.val_sampler, 'set_epoch'):
            self.val_sampler.set_epoch(0)  # Fixed for validation
        
        # Collect local results
        local_preds = []
        local_targets = []
        local_loss = 0.0
        local_samples = 0
        
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc="Validating", disable=not self.is_master)
            
            for batch in iterator:
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
                targets = batch['target']
                
                # Handle string targets
                if isinstance(targets[0], str):
                    class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                else:
                    targets = targets.to(self.device)
                
                outputs = model_for_val(frames)
                loss = self.criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)
                
                # Store local results
                local_preds.extend(preds.cpu().numpy())
                local_targets.extend(targets.cpu().numpy())
                local_loss += loss.item() * targets.size(0)
                local_samples += targets.size(0)
                
                del frames, targets, outputs, loss
        
        if not self.distributed:
            # Single GPU - regular calculation
            return self._calculate_metrics(local_preds, local_targets, local_loss, local_samples)
        
        # Distributed - gather from all GPUs and calculate on complete dataset
        return self._gather_and_calculate_metrics(local_preds, local_targets, local_loss, local_samples)

    def _gather_and_calculate_metrics(self, local_preds, local_targets, local_loss, local_samples):
        """
        Helper function to gather results from all GPUs and calculate metrics
        PyTorch all_gather does all the heavy lifting!
        """
        # Convert to tensors for all_gather
        local_preds_tensor = torch.tensor(local_preds, dtype=torch.long, device=self.device)
        local_targets_tensor = torch.tensor(local_targets, dtype=torch.long, device=self.device)
        local_loss_tensor = torch.tensor(local_loss, dtype=torch.float32, device=self.device)
        local_samples_tensor = torch.tensor(local_samples, dtype=torch.long, device=self.device)
        
        # Prepare to receive results from all GPUs
        world_size = self.world_size
        
        # all_gather requires all tensors to be the same size
        # So first collect the sizes
        all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        size_tensor = torch.tensor([len(local_preds)], dtype=torch.long, device=self.device)
        dist.all_gather(all_sizes, size_tensor)
        
        # Find maximum size
        max_size = max([size.item() for size in all_sizes])
        
        # Pad to maximum size
        if len(local_preds) < max_size:
            pad_size = max_size - len(local_preds)
            local_preds_tensor = torch.cat([
                local_preds_tensor, 
                torch.zeros(pad_size, dtype=torch.long, device=self.device)
            ])
            local_targets_tensor = torch.cat([
                local_targets_tensor,
                torch.zeros(pad_size, dtype=torch.long, device=self.device)
            ])
        
        # Gather predictions and targets
        all_preds = [torch.zeros(max_size, dtype=torch.long, device=self.device) for _ in range(world_size)]
        all_targets = [torch.zeros(max_size, dtype=torch.long, device=self.device) for _ in range(world_size)]
        
        dist.all_gather(all_preds, local_preds_tensor)
        dist.all_gather(all_targets, local_targets_tensor)
        
        # Gather loss and samples (simpler - scalar values)
        all_losses = [torch.zeros(1, dtype=torch.float32, device=self.device) for _ in range(world_size)]
        all_samples_list = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(world_size)]
        
        dist.all_gather(all_losses, local_loss_tensor.unsqueeze(0))
        dist.all_gather(all_samples_list, local_samples_tensor.unsqueeze(0))
        
        # Calculate final metrics - all GPUs compute but result is identical
        final_preds = []
        final_targets = []
        total_loss = 0.0
        total_samples = 0
        
        for i in range(world_size):
            actual_size = all_sizes[i].item()
            final_preds.extend(all_preds[i][:actual_size].cpu().numpy())
            final_targets.extend(all_targets[i][:actual_size].cpu().numpy())
            total_loss += all_losses[i].item()
            total_samples += all_samples_list[i].item()
        
        return self._calculate_metrics(final_preds, final_targets, total_loss, total_samples)

    def _calculate_metrics(self, preds, targets, total_loss, total_samples):
        """
        Helper function to calculate metrics - shared between single and distributed
        """
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = np.mean(np.array(preds) == np.array(targets))
        
        # Per-class precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0, labels=[0, 1, 2]
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
        
        # Log results (all GPUs can log since results are identical)
        if self.is_master:
            self.log(f"Validation Results (DISTRIBUTED):")
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
import pandas as pd

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
   parser.add_argument('--metadata-csv', type=str, default="df_encord_v3.csv",
                      help='Metadata CSV file path')
   parser.add_argument('--video-path-column', type=str, default='video_path',
                      help='Column name containing video file paths')
   parser.add_argument('--label-column', type=str, default='video_type',
                      help='Column name containing labels')
   parser.add_argument('--id-column', type=str, default='id',
                      help='Column name containing video IDs')
   parser.add_argument('--split-column', type=str, default='split',
                      help='Column name containing split info (train/val/test)')
   parser.add_argument('--sample-strategy', type=str, default='metadata_center',
                      choices=['center', 'random', 'metadata_center'],
                      help='Sampling strategy for video frames')
   # parser.add_argument('--center-time-column', type=str, default=None,
   #                    help='Column name for center time (for metadata_center strategy)')
   parser.add_argument('--center-time-column', type=str, default='event_time_sec',
                      help='Column name for center time (for metadata_center strategy)')

   
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
   
   # Video parameters
   parser.add_argument('--fps', type=int, default=10,
                      help='Frames per second to extract')
   parser.add_argument('--duration', type=int, default=5,
                      help='Duration in seconds to extract')
   
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
   
   # Load metadata DataFrame
   metadata_df = pd.read_csv(args.metadata_csv)
   log_info(f"Loaded metadata with {len(metadata_df)} rows")
   
   # Create transforms (try to import, if not available use None)
   try:
       from nexar_video_aug import create_video_transforms
       transform_train  = create_video_transforms(
            mode='train',
            enable_custom_augmentation=True,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            saturation_range=(0.9, 1.1),
            hue_range=(-0.05, 0.05),
            rotation_range=(-7, 7),
            scale_range=(0.95, 1.1),
            translate_range=(0.0, 0.07),
            grayscale_prob=0.02,
            blur_sigma=0.5,
            cutout_prob=0.1,
            cutout_count=(1, 2),
            cutout_size_range=(0.1, 0.15),
            horizontal_flip_prob=0.5,
            aug_probability=0.9
        )
       transform_val = create_video_transforms(mode='val')
       log_info("Using video transforms")
   except ImportError:
       transform_train, transform_val = None, None
       log_info("No video transforms available - using default")
   
   # Create datasets using the new API
   train_data, val_data, test_data = create_datasets_with_manual_split(
       metadata_df=metadata_df,
       video_path_column=args.video_path_column,
       label_column=args.label_column,
       id_column=args.id_column,
       split_column=args.split_column,
       transform_train=transform_train,
       transform_val=transform_val,
       sample_strategy=args.sample_strategy,
       center_time_column=args.center_time_column,
       # VideoDataset parameters
       fps=args.fps,
       duration=args.duration,
   )
   
   log_info(f"Dataset sizes:")
   log_info(f"  Train: {len(train_data) if train_data else 0}")
   log_info(f"  Validation: {len(val_data) if val_data else 0}")
   log_info(f"  Test: {len(test_data) if test_data else 0}")
   
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
   metadata_csv = "df_encord_v3.csv"
   base_model = "convnext_tiny"
   temporal_mode = "gru"
   batch_size = 8
   learning_rate = 1e-4
   epochs = 15
   experiment_name = f"{base_model}_{temporal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
   save_dir = "model_results"
   
   print("Creating datasets...")
   
   # Load metadata DataFrame
   metadata_df = pd.read_csv(metadata_csv)
   print(f"Loaded metadata with {len(metadata_df)} rows")
   
   # Create transforms (try to import, if not available use None)
   try:
       from nexar_video_aug import create_video_transforms
       transform_train = create_video_transforms(
           mode='train',
           enable_custom_augmentation=True,
           brightness_range=(0.95, 1.05),
           contrast_range=(0.95, 1.05),
           saturation_range=(0.95, 1.05),
           hue_range=(-0.02, 0.02),
           rotation_range=(-3, 3),
           scale_range=(0.98, 1.02),
           horizontal_flip_prob=0.5,
           aug_probability=0.8,
       )
       transform_val = create_video_transforms(mode='val')
       print("Using video transforms")
   except ImportError:
       transform_train, transform_val = None, None
       print("No video transforms available - using default")
   
   # Create datasets using the new API
   train_data, val_data, test_data = create_datasets_with_manual_split(
       metadata_df=metadata_df,
       video_path_column='video_path',
       label_column='video_type',
       id_column='id',
       split_column='split',
       transform_train=transform_train,
       transform_val=transform_val,
       sample_strategy='metadata_center',
       fps=10,
       duration=5
   )
   
   print(f"Dataset sizes: Train={len(train_data) if train_data else 0}, Val={len(val_data) if val_data else 0}, Test={len(test_data) if test_data else 0}")
   
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
       metadata_csv = "df_encord_v3.csv"
       video_path_column = 'video_path'
       label_column = 'video_type'
       id_column = 'id'
       split_column = 'split'
       sample_strategy = 'metadata_center'
       center_time_column = None
       base_model = 'resnet18'
       temporal_mode = 'attention'
       epochs = 2  # Just 2 epochs for quick test
       batch_size = 4
       learning_rate = 1e-4
       fps = 10
       duration = 5
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