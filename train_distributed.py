#!/usr/bin/env python3
"""
Distributed Training Script for VideoClassifier

This script demonstrates how to run distributed training on multiple GPUs.

Usage:
    # Single GPU (normal mode):
    python train_distributed.py

    # Multiple GPUs (distributed mode):
    torchrun --nproc_per_node=8 train_distributed.py

    # Multiple GPUs on multiple nodes:
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=<master_node_ip>:29400 train_distributed.py
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Import your VideoClassifier and datasets
from distributed_video_classifier import VideoClassifier, setup_for_distributed_training, is_main_process
from distributed_training_visualizer import create_distributed_visualizer

# Import your dataset classes (replace with your actual imports)
# from your_dataset_module import VideoDataset, create_datasets


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed Video Classification Training')
    
    # Model parameters
    parser.add_argument('--base-model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'resnet101'],
                       help='Base model architecture')
    parser.add_argument('--temporal-mode', type=str, default='attention',
                       choices=['attention', 'lstm', 'gru', 'pooling', 'convolution'],
                       help='Temporal aggregation mode')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of classes')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers per GPU')
    
    # Distributed training parameters
    parser.add_argument('--backend', type=str, default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Distributed backend')
    
    # Visualization parameters
    parser.add_argument('--disable-viz', action='store_true',
                       help='Disable dynamic visualization')
    parser.add_argument('--disable-mini-validation', action='store_true',
                       help='Disable mini-validation during training (not recommended)')
    parser.add_argument('--validation-freq', type=int, default=5,
                       help='Number of validation runs per epoch')
    parser.add_argument('--viz-update-freq', type=int, default=20,
                       help='Update visualization every N mini-batches')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='model_checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    return parser.parse_args()


def create_datasets(data_dir):
    """
    Create train, validation, and test datasets
    
    REPLACE THIS FUNCTION WITH YOUR ACTUAL DATASET CREATION CODE
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # This is a placeholder - replace with your actual dataset creation
    print(f"Creating datasets from {data_dir}")
    
    # Example pseudo-code (replace with your implementation):
    # train_dataset = VideoDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    # val_dataset = VideoDataset(os.path.join(data_dir, 'val'), transform=val_transform)
    # test_dataset = VideoDataset(os.path.join(data_dir, 'test'), transform=test_transform)
    
    # For demonstration purposes, return None (you need to implement this)
    raise NotImplementedError(
        "You need to implement the create_datasets function with your actual dataset creation code. "
        "This should return (train_dataset, val_dataset, test_dataset)"
    )


def setup_distributed_environment():
    """Setup distributed training environment"""
    distributed, local_rank, world_size = setup_for_distributed_training()
    
    if distributed:
        print(f"Distributed training setup: Rank {local_rank}/{world_size}")
        # Set CUDA device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        print("Single GPU/CPU training mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
    
    return distributed, local_rank, world_size, device


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed training
    distributed, local_rank, world_size, device = setup_distributed_environment()
    
    # Only print on main process to avoid spam
    if is_main_process():
        print("=" * 80)
        print("DISTRIBUTED VIDEO CLASSIFICATION TRAINING")
        print("=" * 80)
        print(f"Distributed: {distributed}")
        print(f"World Size: {world_size}")
        print(f"Local Rank: {local_rank}")
        print(f"Device: {device}")
        print(f"Arguments: {args}")
        print("=" * 80)
    
    try:
        # Create datasets
        if is_main_process():
            print("Creating datasets...")
        
        train_dataset, val_dataset, test_dataset = create_datasets(args.data_dir)
        
        if is_main_process():
            print(f"Dataset sizes:")
            print(f"  Train: {len(train_dataset)}")
            print(f"  Validation: {len(val_dataset)}")
            print(f"  Test: {len(test_dataset)}")
        
        # Calculate class weights if needed (optional)
        class_weights = None  # You can implement class weight calculation here
        
        # Create the classifier with distributed support
        if is_main_process():
            print("Creating VideoClassifier...")
        
        classifier = VideoClassifier(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            base_model=args.base_model,
            temporal_mode=args.temporal_mode,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            experiment_name=args.experiment_name,
            device=device,
            num_workers=args.num_workers,
            class_weights=class_weights,
            use_dynamic_viz=not args.disable_viz,
            enable_mini_validation=not args.disable_mini_validation,
            validation_freq=args.validation_freq,
            viz_update_freq=args.viz_update_freq,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            backend=args.backend
        )
        
        if is_main_process():
            print("VideoClassifier created successfully")
            if distributed:
                effective_batch_size = args.batch_size * world_size
                print(f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size} × world_size={world_size})")
        
        # Start training
        if is_main_process():
            print("Starting training...")
        
        history = classifier.train(
            epochs=args.epochs,
            patience=args.patience,
            mixed_precision=True
        )
        
        # Test the model
        if is_main_process():
            print("Starting testing...")
        
        test_results = classifier.test()
        
        # Create visualizations (only on master process)
        if is_main_process():
            print("Creating visualizations...")
            try:
                classifier.visualize_predictions(num_samples=5)
                classifier.visualize_attention_weights()
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
        
        # Print final results
        if is_main_process():
            print("=" * 80)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            if test_results:
                print("Final Test Results:")
                for key, value in test_results.items():
                    if not isinstance(value, dict):
                        print(f"  {key}: {value:.4f}")
            print("=" * 80)
    
    except Exception as e:
        if is_main_process():
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        # Clean up distributed resources
        if distributed:
            classifier.cleanup_distributed()
            if is_main_process():
                print("Distributed resources cleaned up")


def run_single_gpu_example():
    """
    Example function showing how to use the classifier in single GPU mode
    (for Jupyter notebooks or single GPU training)
    """
    print("Running single GPU example...")
    
    # This would be your normal usage in a Jupyter notebook
    # No need to call setup_for_distributed_training() or torchrun
    
    # Create datasets (implement this function)
    # train_dataset, val_dataset, test_dataset = create_datasets("path/to/your/data")
    
    # Create classifier (will automatically detect single GPU mode)
    # classifier = VideoClassifier(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     test_dataset=test_dataset,
    #     base_model='resnet18',
    #     temporal_mode='attention',
    #     num_classes=3,
    #     batch_size=8,
    #     learning_rate=1e-4,
    #     use_dynamic_viz=True  # Will work in Jupyter
    # )
    
    # Train the model
    # history = classifier.train(epochs=30)
    
    # Test the model
    # test_results = classifier.test()
    
    # Create visualizations
    # classifier.visualize_predictions()
    # classifier.visualize_attention_weights()
    
    print("Single GPU example would run here (datasets not implemented)")


if __name__ == "__main__":
    # Check if we're running the script directly
    if len(sys.argv) > 1 and sys.argv[1] == "--single-gpu-example":
        run_single_gpu_example()
    else:
        main()