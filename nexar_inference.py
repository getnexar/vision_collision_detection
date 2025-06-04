import torch
import numpy as np
import cv2
import os
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
from collections import OrderedDict
import tempfile
import shutil


class VideoCollisionModel:
    """
    A class for loading models and analyzing videos for collision detection.
    All predictions exclusively use the dataset approach for consistency and accuracy.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the VideoCollisionModel class.
        
        Args:
            model_path: Path to the model checkpoint (optional, can be loaded later)
        """
        self.model = None
        self.hyperparams = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a model from a checkpoint file
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            model, hyperparams: Loaded model and its hyperparameters
        """
        print(f"Using device: {self.device}")

        try:
            print(f"Attempting to load model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False, pickle_module=None)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        if 'hyperparams' in checkpoint:
            self.hyperparams = checkpoint['hyperparams']
        else:
            self.hyperparams = {
                'base_model': 'convnext_tiny',
                'temporal_mode': 'gru',
                'num_classes': 3
            }

        print("Model hyperparameters:")
        for k, v in self.hyperparams.items():
            print(f"  {k}: {v}")

        try:
            from nexar_arch import EnhancedFrameCNN

            self.model = EnhancedFrameCNN(
                base_model=self.hyperparams['base_model'],
                pretrained=False,
                dropout_rate=0.5,
                temporal_mode=self.hyperparams['temporal_mode'],
                store_attention_weights=True
            )

            feature_dim = self.model.classifier[-1].in_features
            num_classes = self.hyperparams['num_classes']
            self.model.classifier[-1] = torch.nn.Linear(feature_dim, num_classes)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(state_dict)
                print("Model weights loaded successfully")
            else:
                print("Warning: No weights found in checkpoint")

            self.model.to(self.device)
            self.model.eval()
            return self.model, self.hyperparams

        except ImportError as e:
            print(f"Model class import error: {e}")
            print("Please ensure 'nexar_arch' module is available")
            return None, None

        except Exception as e:
            print(f"Model creation error: {e}")
            return None, None

    def predict(self, video_paths, batch_size=8, num_workers=1, sample_strategy='center', verbose=True):
        """
        Run inference on video files using the dataset approach.
        This is the only method to perform prediction in this class.
        
        Args:
            video_paths: Path to a single video or list of paths to video files
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
            sample_strategy: Strategy for sampling frames - 'random', 'metadata_time', or 'center'
            verbose: Whether to show progress
            
        Returns:
            For a single video: Dictionary with prediction results
            For multiple videos: List of dictionaries with prediction results
        """
        # Check if model is loaded
        if self.model is None:
            print("Error: Model not loaded. Please load a model first.")
            return None
        
        # Convert single video path to list if needed
        single_video = False
        if isinstance(video_paths, str):
            video_paths = [video_paths]
            single_video = True
        
        # Validate that the video files exist
        valid_paths = []
        for path in video_paths:
            if not os.path.exists(path):
                print(f"Warning: Video file not found: {path}")
                continue
            valid_paths.append(path)
        
        if not valid_paths:
            print("Error: No valid video files found")
            return [] if not single_video else None
            
        video_paths = valid_paths
        
        # If a directory is provided, get all video files in it
        if len(video_paths) == 1 and os.path.isdir(video_paths[0]):
            directory = video_paths[0]
            video_paths = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.mp4', '.mov', '.avi')):
                        video_paths.append(os.path.join(root, file))
            if verbose:
                print(f"Found {len(video_paths)} videos in directory {directory}")
            
            if not video_paths:
                print(f"Error: No video files found in directory {directory}")
                return [] if not single_video else None
        
        # Get original filenames for reference
        original_filenames = [Path(path).name for path in video_paths]
        
        # Create temporary metadata DataFrame for the dataset
        metadata = pd.DataFrame({
            'id': [f"video_{i}" for i in range(len(video_paths))],  # Use consistent IDs
            'video_type': ['Normal'] * len(video_paths)  # Default value, will be predicted
        })
        
        # Create a temporary directory structure expected by the dataset
        temp_dir = tempfile.mkdtemp()
        
        # Create a mapping from dataset index to original video path
        index_to_path = {}
        
        try:
            # Copy or symlink files to the temp directory with the expected structure
            for i, path in enumerate(video_paths):
                video_id = f"video_{i}"
                
                # Create video directory
                video_dir = os.path.join(temp_dir, video_id)
                os.makedirs(video_dir, exist_ok=True)
                
                # Link or copy the video file
                dest_path = os.path.join(video_dir, f"{video_id}.mp4")
                if os.path.islink(dest_path):
                    os.unlink(dest_path)
                    
                # On Windows, we need to copy the file; on Linux/Mac we can use symlinks
                if os.name == 'nt':
                    shutil.copy2(path, dest_path)
                else:
                    try:
                        os.symlink(os.path.abspath(path), dest_path)
                    except:
                        # If symlink fails, fallback to copy
                        shutil.copy2(path, dest_path)
                
                # Store the mapping
                index_to_path[i] = path
            
            # Create the inference dataset
            try:
                from nexar_video_aug import create_video_transforms
                from nexar_data import NvidiaDashcamDataset
            except ImportError as e:
                print(f"Required package import error: {e}")
                print("Cannot run prediction without required packages.")
                print("Please ensure 'nexar_video_aug' and 'nexar_data' modules are available")
                return [] if not single_video else None
            
            inference_dataset = NvidiaDashcamDataset(
                metadata_df=metadata,
                base_dirs=[temp_dir],
                fps=10,            # Same as training
                duration=5,        # Same as training
                is_train=False,    # Important: use validation/test mode
                skip_missing=False,
                transform=create_video_transforms(mode='val'),  # Use validation transforms
                sample_strategy=sample_strategy
            )
            
            # Create a DataLoader
            from torch.utils.data import DataLoader
            
            inference_loader = DataLoader(
                inference_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            # Run inference
            all_results = []
            self.model.eval()
            
            # Set up tracking for processed items
            processed_indices = []
            
            # Get hyperparameters for classification
            hyperparams = self.hyperparams or {'num_classes': 3}
            num_classes = hyperparams.get('num_classes', 3)
            class_map = {0: 'Normal', 1: 'Near Collision', 2: 'Collision'}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(inference_loader, desc="Predicting", disable=not verbose)):
                    # Get data
                    frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
                    batch_ids = batch['id']
                    
                    # Forward pass
                    outputs = self.model(frames)
                    
                    # Get predictions and probabilities
                    if num_classes == 2:
                        # Binary classification
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        if probs.ndim == 1:
                            # Handle single output case
                            probs = np.column_stack((1 - probs, probs))
                        pred_classes = (probs > 0.5).astype(int)
                        if pred_classes.ndim == 1:
                            pred_classes = pred_classes[:, 1]
                    else:
                        # Multi-class classification
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    # Calculate batch start index
                    batch_start = batch_idx * batch_size
                    
                    # Create result dictionaries for each video in the batch
                    for i in range(len(batch_ids)):
                        # Get the dataset index for this item
                        dataset_idx = batch_start + i
                        
                        # Get original video path
                        if hasattr(inference_dataset, 'valid_indices') and len(inference_dataset.valid_indices) > 0:
                            # If dataset provides valid indices mapping
                            original_idx = inference_dataset.valid_indices[dataset_idx] if dataset_idx < len(inference_dataset.valid_indices) else dataset_idx
                        else:
                            # Use direct mapping
                            original_idx = dataset_idx
                        
                        # Skip if we've somehow gone past the end of our mapping
                        if original_idx >= len(video_paths):
                            continue
                            
                        # Get the original path
                        original_path = index_to_path.get(original_idx, video_paths[original_idx])
                        original_filename = original_filenames[original_idx]
                        
                        # Get the prediction class
                        pred_class = pred_classes[i] if i < len(pred_classes) else 0
                        pred_class_name = class_map.get(pred_class, f"Class {pred_class}")
                        
                        # Create probabilities dict
                        prob_dict = {}
                        for k in range(num_classes):
                            class_key = class_map.get(k, f"Class {k}")
                            prob_value = float(probs[i, k]) if i < len(probs) and k < probs.shape[1] else 0.0
                            prob_dict[class_key] = prob_value
                        
                        # Create results dictionary
                        results = {
                            'predicted_class': int(pred_class),
                            'predicted_class_name': pred_class_name,
                            'probabilities': prob_dict,
                            'video_path': original_path,
                            'filename': original_filename
                        }
                        
                        # Store metadata about frame sampling
                        if sample_strategy == 'center':
                            video_cap = cv2.VideoCapture(original_path)
                            if video_cap.isOpened():
                                frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                video_cap.release()
                                results['center_frame'] = frame_count // 2
                        
                        all_results.append(results)
                        processed_indices.append(original_idx)
            
            # Check if all videos were processed
            if len(processed_indices) != len(video_paths):
                missing = set(range(len(video_paths))) - set(processed_indices)
                if verbose and missing:
                    print(f"Warning: {len(missing)} videos were not processed")
            
            # Return single result for single video input or list for multiple videos
            if single_video and all_results and len(all_results) > 0:
                return all_results[0]
            return all_results
        
        finally:
            # Clean up the temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")

    def display_results(self, results, use_widget=False):
        """
        Display inference results with a visual bar or fancy widget
        
        Args:
            results: Dictionary or list of dictionaries with prediction results
            use_widget: Whether to use a fancy widget (default: False)
        """
        if results is None:
            print("No results to display")
            return

        # Handle both single result and list of results
        if isinstance(results, list):
            for result in results:
                if 'video_path' in result:
                    print(f"\n==== Results for {result['video_path']} ====")
                self._display_single_result(result, use_widget)
        else:
            self._display_single_result(results, use_widget)

    def _display_single_result(self, results, use_widget=False):
        """
        Display a single inference result
        
        Args:
            results: Dictionary with prediction results
            use_widget: Whether to use a fancy widget (default: False)
        """
        if use_widget:
            self._display_fancy_widget(results)
        else:
            self._display_simple_results(results)

    def _display_simple_results(self, results):
        """
        Display inference results with a simple visual bar
        
        Args:
            results: Dictionary with prediction results
        """
        print("\n==== Analysis Results ====")
        print(f"Predicted class: {results['predicted_class_name']}")
        
        if 'center_frame' in results and results['center_frame'] is not None:
            print(f"Centered around frame: {results['center_frame']}")
        
        print("\nProbabilities:")

        # Sort probabilities by value (descending)
        sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)

        for class_name, prob in sorted_probs:
            # Create a visual bar
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)

            # Set color based on class
            if class_name == "Normal":
                color = "\033[92m"  # Green
            elif class_name == "Near Collision":
                color = "\033[93m"  # Yellow
            else:
                color = "\033[91m"  # Red
            reset = "\033[0m"  # Reset color

            print(f"  {color}{class_name:<13}{reset}: {prob:.4f} {bar} {prob*100:.1f}%")

    def _display_fancy_widget(self, results):
        """
        Display a fancy widget for the results (for Jupyter notebooks)
        
        Args:
            results: Dictionary with prediction results
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib import patches
            from IPython.display import display, HTML
            import ipywidgets as widgets
            
            # Sort probabilities by value (descending)
            sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
            
            # Colors for classes
            color_map = {
                "Normal": "#4CAF50",       # Green
                "Near Collision": "#FF9800", # Orange
                "Collision": "#F44336"      # Red
            }
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Background style
            ax.set_facecolor('#F5F5F5')
            
            # Title
            ax.text(0.5, 0.9, "Video Analysis Results", 
                    horizontalalignment='center',
                    fontsize=18, fontweight='bold')
            
            # Predicted class indicator
            pred_class = results['predicted_class_name']
            ax.text(0.5, 0.82, f"Predicted: {pred_class}",
                    horizontalalignment='center',
                    fontsize=16, fontweight='bold',
                    color=color_map.get(pred_class, '#333333'))
            
            # Draw probability bars
            y_pos = 0.7
            bar_height = 0.08
            text_offset = 0.02
            
            for i, (class_name, prob) in enumerate(sorted_probs):
                color = color_map.get(class_name, '#999999')
                
                # Main bar (background)
                ax.add_patch(patches.Rectangle(
                    (0.15, y_pos - bar_height/2), 0.7, bar_height,
                    facecolor='#E0E0E0', edgecolor='none', alpha=0.5))
                
                # Progress bar
                ax.add_patch(patches.Rectangle(
                    (0.15, y_pos - bar_height/2), max(0.01, prob * 0.7), bar_height,
                    facecolor=color, edgecolor='none'))
                
                # Class name
                ax.text(0.13, y_pos, class_name, 
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=12, fontweight='bold')
                
                # Percentage
                ax.text(0.86, y_pos, f"{prob * 100:.1f}%", 
                        horizontalalignment='left',
                        verticalalignment='center', 
                        fontsize=12)
                
                # Move y position for next bar
                y_pos -= 0.15
            
            # Add some metadata at the bottom if available
            meta_text = ""
            if 'video_path' in results:
                meta_text += f"File: {Path(results['video_path']).name}"
            if 'center_frame' in results and results['center_frame'] is not None:
                meta_text += f" | Centered around frame: {results['center_frame']}"
                
            if meta_text:
                ax.text(0.5, 0.1, meta_text,
                        horizontalalignment='center',
                        fontsize=10, color='#666666')
            
            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Display the figure
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"Couldn't display fancy widget, missing dependencies: {e}")
            # Fall back to simple display
            self._display_simple_results(results)