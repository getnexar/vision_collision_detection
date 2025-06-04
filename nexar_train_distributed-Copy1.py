#!/usr/bin/env python3
"""
Nexar Video Classification - Distributed Training Script

This script converts your Jupyter notebook workflow to a command-line script
that works with both single GPU and distributed multi-GPU training.

Usage:
    # Single GPU (like your notebook):
    python nexar_train_distributed.py

    # Multiple GPUs:
    torchrun --nproc_per_node=8 nexar_train_distributed.py

    # With custom parameters:
    python nexar_train_distributed.py --base-model resnet18 --temporal-mode attention --epochs 30
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

# Import your existing modules
try:
    from nexar_train import *
    from nexar_videos import *
    from distributed_video_classifier import VideoClassifier, setup_for_distributed_training, is_main_process
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure nexar_train.py, nexar_videos.py, and distributed_video_classifier.py are in the same directory")
    sys.exit(1)


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
    parser = argparse.ArgumentParser(description='Nexar Video Classification - Distributed Training')
    
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
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Experiment parameters
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save-dir', type=str, default='model_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Distributed training parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers per GPU')
    parser.add_argument('--disable-viz', action='store_true',
                       help='Disable dynamic visualization')
    parser.add_argument('--disable-mini-validation', action='store_true',
                       help='Disable mini-validation during training')
    parser.add_argument('--validation-freq', type=int, default=5,
                       help='Number of validation runs per epoch')
    
    # Grid search
    parser.add_argument('--run-grid-search', action='store_true',
                       help='Run grid search instead of single model training')
    
    return parser.parse_args()


def create_experiment_name(base_model, temporal_mode, timestamp=None):
    """Create experiment name like in your notebook"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_model}_{temporal_mode}_{timestamp}"


def setup_distributed_environment():
    """Setup distributed training environment"""
    distributed, local_rank, world_size = setup_for_distributed_training()
    
    if distributed:
        if is_main_process():
            print(f"Distributed training setup: Rank {local_rank}/{world_size}")
        # Set CUDA device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if is_main_process():
            print("Single GPU/CPU training mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
    
    return distributed, local_rank, world_size, device


def run_single_experiment(args, distributed, local_rank, world_size, device):
    """Run a single training experiment (equivalent to your notebook code)"""
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = create_experiment_name(args.base_model, args.temporal_mode)
    
    if is_main_process():
        print("=" * 80)
        print("NEXAR VIDEO CLASSIFICATION - DISTRIBUTED TRAINING")
        print("=" * 80)
        print(f"Experiment: {args.experiment_name}")
        print(f"Base Model: {args.base_model}")
        print(f"Temporal Mode: {args.temporal_mode}")
        print(f"Distributed: {distributed}")
        if distributed:
            print(f"World Size: {world_size}")
            print(f"Effective Batch Size: {args.batch_size * world_size}")
        print("=" * 80)
    
    # Create save directory
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets (same as your notebook)
    if is_main_process():
        print("Creating datasets...")
    
    train_data, val_data, test_data = create_datasets_with_manual_split(
        base_dirs=args.base_dirs,
        metadata_csv=args.metadata_csv,
        seed=args.seed,
        sensor_subdir=args.sensor_subdir,
        sample_strategy=args.sample_strategy,
        show_stats=is_main_process()  # Only show stats on main process
    )
    
    if is_main_process():
        print(f"Dataset sizes:")
        print(f"  Train: {len(train_data)}")
        print(f"  Validation: {len(val_data)}")
        print(f"  Test: {len(test_data)}")
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights and is_main_process():
        print("Calculating class weights...")
        # You can implement this based on your nexar_train.py logic
        # class_weights = calculate_class_weights(train_data)
    
    # Create the VideoClassifier with distributed support
    if is_main_process():
        print("Creating VideoClassifier...")
    
    classifier = VideoClassifier(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        base_model=args.base_model,
        temporal_mode=args.temporal_mode,
        num_classes=3,  # Normal, Near Collision, Collision
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
        distributed=distributed,
        local_rank=local_rank,
        world_size=world_size
    )
    
    if is_main_process():
        print("Starting training...")
    
    # Train the model
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
            if args.temporal_mode == 'attention':
                classifier.visualize_attention_weights()
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    # Print final results
    if is_main_process():
        print("=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        if test_results:
            print("Final Test Results:")
            for key, value in test_results.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key} per class:")
                    class_names = ['Normal', 'Near Collision', 'Collision']
                    for cls_idx, cls_value in value.items():
                        class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
                        print(f"    {class_name}: {cls_value:.4f}")
        print(f"Results saved in: {args.save_dir}/{args.experiment_name}")
        print("=" * 80)
    
    # Clean up distributed resources
    if distributed:
        classifier.cleanup_distributed()
    
    return classifier, history, test_results


def run_grid_search(args, distributed, local_rank, world_size, device):
    """Run grid search experiments"""
    if is_main_process():
        print("=" * 80)
        print("RUNNING GRID SEARCH")
        print("=" * 80)
    
    # Define grid search parameters (modify as needed)
    base_models = ['resnet18', 'convnext_tiny']
    temporal_modes = ['attention', 'gru', 'lstm']
    learning_rates = [1e-4, 5e-5]
    
    results = []
    
    for base_model in base_models:
        for temporal_mode in temporal_modes:
            for lr in learning_rates:
                if is_main_process():
                    print(f"\nRunning experiment: {base_model} + {temporal_mode} + lr={lr}")
                
                # Update args for this experiment
                args.base_model = base_model
                args.temporal_mode = temporal_mode
                args.learning_rate = lr
                args.experiment_name = create_experiment_name(
                    base_model, temporal_mode, 
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{lr}"
                )
                
                try:
                    classifier, history, test_results = run_single_experiment(
                        args, distributed, local_rank, world_size, device
                    )
                    
                    if is_main_process():
                        results.append({
                            'base_model': base_model,
                            'temporal_mode': temporal_mode,
                            'learning_rate': lr,
                            'experiment_name': args.experiment_name,
                            'test_results': test_results,
                            'best_val_loss': classifier.best_val_loss
                        })
                        
                except Exception as e:
                    if is_main_process():
                        print(f"Error in experiment {base_model}+{temporal_mode}+{lr}: {e}")
                    continue
    
    # Print grid search summary
    if is_main_process():
        print("=" * 80)
        print("GRID SEARCH RESULTS SUMMARY")
        print("=" * 80)
        for result in results:
            print(f"{result['base_model']} + {result['temporal_mode']} + lr={result['learning_rate']}")
            print(f"  Best Val Loss: {result['best_val_loss']:.6f}")
            if result['test_results']:
                print(f"  Test Accuracy: {result['test_results'].get('accuracy', 0):.4f}")
            print(f"  Experiment: {result['experiment_name']}")
            print()
    
    return results


def main():
    """Main function"""
    args = parse_args()
    
    # Setup distributed training
    distributed, local_rank, world_size, device = setup_distributed_environment()
    
    try:
        if args.run_grid_search:
            results = run_grid_search(args, distributed, local_rank, world_size, device)
        else:
            classifier, history, test_results = run_single_experiment(
                args, distributed, local_rank, world_size, device
            )
            
    except Exception as e:
        if is_main_process():
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        raise


def run_notebook_equivalent():
    """
    Function that replicates your exact notebook code for comparison
    Call this from a notebook or interactive Python session
    """
    # Your original notebook code, adapted for function form
    seed = 42
    set_random_seeds(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define parameters (same as your notebook)
    base_dirs = ["../data/research-nvidia-data/nvidia-1", "../data/research-nvidia-data/nvidia-2"]
    metadata_csv = "nvidia_delivery_to_train.csv"
    base_model = "convnext_tiny"
    temporal_mode = "gru"
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-4
    epochs = 15
    experiment_name = f"{base_model}_{temporal_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = "model_results"
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_data, val_data, test_data = create_datasets_with_manual_split(
        base_dirs=base_dirs,
        metadata_csv=metadata_csv,
        seed=seed,
        sensor_subdir='signals',
        sample_strategy='center',
        show_stats=True
    )
    
    # Option 1: Use your original run_experiment function
    # classifier = run_experiment(
    #     train_data=train_data,
    #     val_data=val_data,
    #     test_data=test_data,
    #     base_model=base_model,
    #     temporal_mode=temporal_mode,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     experiment_name=experiment_name,
    #     use_class_weights=True,
    #     save_dir=save_dir
    # )
    
    # Option 2: Use the new distributed VideoClassifier (will work in single GPU mode)
    classifier = VideoClassifier(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        base_model=base_model,
        temporal_mode=temporal_mode,
        num_classes=3,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=save_dir,
        experiment_name=experiment_name,
        device=device,
        use_dynamic_viz=True,  # Will work in Jupyter
        # distributed=False is auto-detected
    )
    
    # Train and test
    history = classifier.train(epochs=epochs)
    test_results = classifier.test()
    
    return classifier, history, test_results


if __name__ == "__main__":
    main()