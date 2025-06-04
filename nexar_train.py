import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output, Javascript
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from IPython.display import display, HTML


class DynamicTrainingVisualizer:
    """
    An improved class to visualize training metrics dynamically during training with
    persistent display and efficient updates. Shows moving average for smoother visualization.
    Now includes full validation results tracking.
    """
    def __init__(self, max_loss=None, num_epochs=None, num_iterations_per_epoch=None, 
                 update_freq=20, num_classes=3, class_names=None, moving_avg_window=29):
        """
        Initialize the visualizer
        
        Args:
            max_loss: Maximum expected loss for y-axis scaling, will be auto-adjusted if None
            num_epochs: Total number of epochs for the training process
            num_iterations_per_epoch: Number of iterations per epoch
            update_freq: Update visualization every N iterations
            num_classes: Number of classes in the classification task
            class_names: Names of the classes, defaults to ['Normal', 'Near Collision', 'Collision']
            moving_avg_window: Window size for moving average calculation of training loss
        """
        self.max_loss = max_loss
        self.num_epochs = num_epochs
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.update_freq = update_freq
        self.num_classes = num_classes
        self.class_names = class_names if class_names else ['Normal', 'Near Collision', 'Collision']
        self.moving_avg_window = moving_avg_window
        
        # Initialize metrics storage
        self.train_losses = []
        self.train_losses_ma = []  # Moving average of train losses
        self.val_losses = []
        self.iterations = []
        self.val_iterations = []
        self.current_epoch = 0
        self.current_iteration = 0
        self.epoch_boundaries = []
        
        # Full validation tracking
        self.full_val_losses = []
        self.full_val_iterations = []
        self.full_val_metrics_history = {
            'accuracy': [],
            'precision': {i: [] for i in range(num_classes)},
            'recall': {i: [] for i in range(num_classes)},
            'f1': {i: [] for i in range(num_classes)}
        }
        
        # Tracking for metrics over time
        self.metrics_history = {
            'accuracy': [],
            'precision': {i: [] for i in range(num_classes)},
            'recall': {i: [] for i in range(num_classes)},
            'f1': {i: [] for i in range(num_classes)}
        }
        
        # Current best metrics
        self.best_metrics = {
            'accuracy': 0.0,
            'precision': {i: 0.0 for i in range(num_classes)},
            'recall': {i: 0.0 for i in range(num_classes)},
            'f1': {i: 0.0 for i in range(num_classes)}
        }
        
        # Time tracking
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.epoch_start_time = None
        
        # For persistent display
        self.output_html = None
        self.is_initialized = False
        
        # For animation
        self.animation = None
        
        # Create CSS for styling
        self.setup_css()
    
    def _calculate_moving_average(self, values, window_size):
        """Calculate moving average with specified window size"""
        if len(values) < 2:
            return values
        
        # Calculate the moving average
        ma = []
        for i in range(len(values)):
            if i < window_size - 1:
                # For the first window_size-1 points, take available points
                window = values[:i+1]
            else:
                # Once we have enough points, use window_size points
                window = values[i-window_size+1:i+1]
            
            ma.append(sum(window) / len(window))
        
        return ma
    
    def setup_css(self):
        """Set up CSS styles for better visualization"""
        self.css = """
        <style>
            .training-dashboard {
                font-family: Arial, sans-serif;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background-color: #f8f9fa;
            }
            
            .progress-container {
                margin: 10px 0;
                background-color: #e9ecef;
                border-radius: 4px;
                position: relative;
            }
            
            .progress-bar {
                height: 25px;
                background-color: #007bff;
                border-radius: 4px;
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
            }
            
            .progress-bar-secondary {
                background-color: #6c757d;
            }
            
            .progress-text {
                position: absolute;
                width: 100%;
                text-align: center;
                font-weight: bold;
                color: #333;
                line-height: 25px;
            }
            
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            
            .metrics-table th, .metrics-table td {
                padding: 8px;
                text-align: center;
                border: 1px solid #dee2e6;
            }
            
            .metrics-table th {
                background-color: #e9ecef;
                font-weight: bold;
            }
            
            .metrics-table tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            
            .metrics-value {
                font-family: monospace;
            }
            
            .time-info {
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
                font-size: 0.9em;
                color: #6c757d;
            }
            
            .best-metric {
                color: #28a745;
                font-weight: bold;
            }
            
            .visualization-container {
                display: flex;
                flex-direction: column;
                margin-top: 15px;
            }
        </style>
        """
        
        # Display the CSS once
        display(HTML(self.css))
    
    def initialize_display(self):
        """Create the initial display structure with placeholders"""
        if self.is_initialized:
            return
                
        # Create the main dashboard container
        dashboard_html = """
        <div class="training-dashboard" id="training-dashboard">
            <h2>Training Progress</h2>
            
            <div>
                <h3>Overall Progress</h3>
                <div class="progress-container">
                    <div class="progress-text" id="epoch-progress-text">Epoch 0/0 (0%)</div>
                    <div class="progress-bar" id="epoch-progress-bar" style="width: 0%"></div>
                </div>
            </div>
            
            <div>
                <h3>Current Epoch Progress</h3>
                <div class="progress-container">
                    <div class="progress-text" id="iteration-progress-text">Iteration 0/0 (0%)</div>
                    <div class="progress-bar progress-bar-secondary" id="iteration-progress-bar" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="visualization-container">
                <div id="plot-container" style="height: 400px;">
                    <!-- Plot will be inserted here -->
                </div>
                
                <div id="metrics-table-container">
                    <!-- Metrics table will be inserted here -->
                </div>
            </div>
            
            <div class="time-info">
                <div id="elapsed-time">Elapsed: 00:00:00</div>
                <div id="eta">ETA: --:--:--</div>
                <div id="iter-per-sec">Iterations/sec: 0.00</div>
            </div>
        </div>
        """
        
        # Display the dashboard
        display(HTML(dashboard_html))
        self.is_initialized = True
        
        # Create the figure for plots (initially empty)
        self._setup_plot()
    
    def _setup_plot(self):
        """Initialize the visualization figure with subplots for loss and metrics"""
        # Create figure with gridspec for flexible layout
        self.fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 2, width_ratios=[1, 1], figure=self.fig)
        
        # Add loss plot on the left
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_loss.set_title(f'Training Loss (MA {self.moving_avg_window}), Validation Loss, and Full Validation Loss')
        self.ax_loss.set_xlabel('Iterations')
        self.ax_loss.set_ylabel('Loss')
        if self.max_loss:
            self.ax_loss.set_ylim(0, self.max_loss)
        
        # Initialize empty plots
        self.train_line, = self.ax_loss.plot([], [], 'b-', label='Training Loss MA', linewidth=2)
        self.val_line, = self.ax_loss.plot([], [], 'r-', label='Mini-Validation Loss', linewidth=2)
        self.full_val_line, = self.ax_loss.plot([], [], 'g-', label='Full Validation Loss', linewidth=2, marker='o')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.grid(True, linestyle='--', alpha=0.7)
        
        # Add metrics visualization on the right
        self.ax_metrics = self.fig.add_subplot(gs[0, 1])
        self.ax_metrics.set_title('Validation Metrics')
        self.ax_metrics.set_xlabel('Iterations')
        self.ax_metrics.set_ylabel('Score')
        self.ax_metrics.set_ylim(0, 1.05)
        
        # Initialize lines for different metrics
        self.acc_line, = self.ax_metrics.plot([], [], 'g-', label='Accuracy (Mini)', linewidth=2)
        self.f1_line, = self.ax_metrics.plot([], [], 'c-', label='F1 Score (Mini)', linewidth=2)
        self.full_acc_line, = self.ax_metrics.plot([], [], 'g--', label='Accuracy (Full)', linewidth=2, marker='o')
        self.full_f1_line, = self.ax_metrics.plot([], [], 'c--', label='F1 Score (Full)', linewidth=2, marker='o')
        
        self.ax_metrics.legend(loc='lower right')
        self.ax_metrics.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the initial empty plot
        plot_filename = 'temp_training_plot.png'
        self.fig.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()  # Close the figure to prevent display
        
        # Update the plot container with the initial image
        self._update_plot_in_dashboard(plot_filename)
    
    def _update_plot_in_dashboard(self, plot_filename):
        """Update the plot in the dashboard with a new image"""
        # Read the image file and convert to base64
        import base64
        with open(plot_filename, 'rb') as f:
            img_data = f.read()
        
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        img_src = f'data:image/png;base64,{img_base64}'
        
        # Use JavaScript to update the plot container
        js_code = f"""
        var plotImg = document.getElementById('training-plot-img');
        if (plotImg) {{
            plotImg.src = '{img_src}';
        }} else {{
            var imgHtml = '<img id="training-plot-img" src="{img_src}" style="width:100%;max-height:400px;">';
            document.getElementById('plot-container').innerHTML = imgHtml;
        }}
        """
        
        display(Javascript(js_code))
        
        # Clean up temp file
        try:
            os.remove(plot_filename)
        except:
            pass
    
    def _create_metrics_table_html(self, latest_metrics=None):
        """Create HTML for the metrics table"""
        if latest_metrics is None:
            latest_metrics = {}
        
        # Start with header row
        table_html = """
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
        """
        
        # Add class names to header
        for class_name in self.class_names:
            table_html += f"<th>{class_name}</th>"
        
        table_html += "<th>Overall</th></tr>"
        
        # Add metric rows
        metrics_rows = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1']
        
        for row_name, metric_key in zip(metrics_rows, metric_keys):
            table_html += f"<tr><td>{row_name}</td>"
            
            # Per-class metrics (if available)
            if metric_key in latest_metrics and isinstance(latest_metrics[metric_key], dict):
                for cls_idx in range(self.num_classes):
                    value = latest_metrics[metric_key].get(cls_idx, 0.0)
                    best_value = self.best_metrics[metric_key].get(cls_idx, 0.0)
                    
                    # Mark as best if it's the best so far
                    if value >= best_value:
                        self.best_metrics[metric_key][cls_idx] = value
                        table_html += f'<td class="metrics-value best-metric">{value:.4f}</td>'
                    else:
                        table_html += f'<td class="metrics-value">{value:.4f}</td>'
            else:
                # Fill with placeholders if not available
                for _ in range(self.num_classes):
                    table_html += '<td class="metrics-value">N/A</td>'
            
            # Overall metric
            if metric_key in latest_metrics:
                if isinstance(latest_metrics[metric_key], dict):
                    # Average the per-class metrics
                    avg_value = np.mean(list(latest_metrics[metric_key].values()))
                    overall_value = avg_value
                else:
                    # Use the overall metric directly
                    overall_value = latest_metrics[metric_key]
                
                # Mark as best if it's the best so far
                if metric_key == 'accuracy' and overall_value >= self.best_metrics[metric_key]:
                    self.best_metrics[metric_key] = overall_value
                    table_html += f'<td class="metrics-value best-metric">{overall_value:.4f}</td>'
                else:
                    table_html += f'<td class="metrics-value">{overall_value:.4f}</td>'
            else:
                table_html += '<td class="metrics-value">N/A</td>'
            
            table_html += "</tr>"
        
        table_html += "</table>"
        return table_html
    
    def _update_metrics_table_in_dashboard(self, latest_metrics=None):
        """Update the metrics table in the dashboard"""
        table_html = self._create_metrics_table_html(latest_metrics)
        
        # Use JavaScript to update the table container
        js_code = f"""
        document.getElementById('metrics-table-container').innerHTML = `{table_html}`;
        """
        display(Javascript(js_code))
    
    def _update_progress_bars(self, force_update=False):
        """Update the progress bars in the dashboard"""
        # Calculate progress percentages
        if self.num_epochs:
            epoch_progress = min(100, (self.current_epoch / self.num_epochs) * 100)
        else:
            epoch_progress = 0
            
        if self.num_iterations_per_epoch:
            # Get iteration within current epoch
            iter_in_epoch = self.current_iteration % self.num_iterations_per_epoch if self.num_iterations_per_epoch else 0
            iter_progress = min(100, (iter_in_epoch / self.num_iterations_per_epoch) * 100)
        else:
            iter_progress = 0
        
        # Use JavaScript to update progress bars
        js_code = f"""
        // Update the overall epoch progress - will persist
        var epochBar = document.getElementById('epoch-progress-bar');
        var epochText = document.getElementById('epoch-progress-text');
        if (epochBar && epochText) {{
            epochBar.style.width = "{epoch_progress}%";
            epochText.innerText = "Epoch {self.current_epoch}/{self.num_epochs} ({epoch_progress:.1f}%)";
        }}
        
        // Update the current epoch iteration progress
        var iterBar = document.getElementById('iteration-progress-bar');
        var iterText = document.getElementById('iteration-progress-text');
        if (iterBar && iterText) {{
            iterBar.style.width = "{iter_progress}%";
            iterText.innerText = "Iteration {iter_in_epoch}/{self.num_iterations_per_epoch} ({iter_progress:.1f}%)";
        }}
        """
        display(Javascript(js_code))
    
    def _update_time_info(self):
        """Update the time information in the dashboard"""
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        
        # Calculate elapsed time string
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Calculate ETA if possible
        eta_str = "--:--:--"
        if self.num_epochs and self.current_epoch > 0:
            progress_fraction = self.current_epoch / self.num_epochs
            if progress_fraction > 0:
                total_estimated = total_elapsed / progress_fraction
                remaining = total_estimated - total_elapsed
                
                # Format remaining time
                hours, remainder = divmod(remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                eta_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Calculate iterations per second
        if self.current_iteration > 0:
            iter_per_sec = self.current_iteration / total_elapsed
        else:
            iter_per_sec = 0.0
        
        # Use JavaScript to update time info
        from IPython.display import Javascript, display
        js_code = f"""
        document.getElementById('elapsed-time').innerText = "Elapsed: {elapsed_str}";
        document.getElementById('eta').innerText = "ETA: {eta_str}";
        document.getElementById('iter-per-sec').innerText = "Iterations/sec: {iter_per_sec:.2f}";
        """
        display(Javascript(js_code))
    
    def start_epoch(self, epoch_num):
        """Mark the start of a new epoch"""
        self.current_epoch = epoch_num
        self.epoch_start_time = time.time()
        
        # Update the progress display
        if self.is_initialized:
            self._update_progress_bars()
            self._update_time_info()
        else:
            self.initialize_display()
    
    def update_train_loss(self, iteration, loss):
        """Update the training loss data with new information"""
        self.current_iteration = iteration
        self.iterations.append(iteration)
        self.train_losses.append(loss)
        
        # Calculate moving average
        self.train_losses_ma = self._calculate_moving_average(
            self.train_losses, self.moving_avg_window)
        
        # Check if visualization needs updating
        if self.is_initialized and (len(self.iterations) % self.update_freq == 0 or len(self.iterations) == 1):
            self._update_plots()
            self._update_progress_bars()
            self._update_time_info()
    
    def update_val_metrics(self, iteration, val_loss, val_metrics):
        """Update validation metrics with new information"""
        self.val_iterations.append(iteration)
        self.val_losses.append(val_loss)
        
        # Update metrics history (no moving average for validation metrics)
        for k, v in val_metrics.items():
            if k == 'accuracy':
                self.metrics_history[k].append(v)
            elif k in ['precision', 'recall', 'f1'] and isinstance(v, dict):
                for class_idx, value in v.items():
                    if class_idx in self.metrics_history[k]:
                        self.metrics_history[k][class_idx].append(value)
        
        # Update the plots and metrics table
        if self.is_initialized:
            self._update_plots()
            self._update_metrics_table_in_dashboard(val_metrics)
            self._update_progress_bars()
            self._update_time_info()
    
    def update_full_val_metrics(self, iteration, val_loss, val_metrics):
        """Update full validation metrics with new information"""
        self.full_val_iterations.append(iteration)
        self.full_val_losses.append(val_loss)
        
        # Update full validation metrics history
        for k, v in val_metrics.items():
            if k == 'accuracy':
                self.full_val_metrics_history[k].append(v)
            elif k in ['precision', 'recall', 'f1'] and isinstance(v, dict):
                for class_idx, value in v.items():
                    if class_idx in self.full_val_metrics_history[k]:
                        self.full_val_metrics_history[k][class_idx].append(value)
        
        # Update plots with full validation results
        if self.is_initialized:
            self._update_plots()
    
    def mark_epoch(self, epoch, iteration):
        """Mark the end of an epoch on the plot"""
        self.epoch_boundaries.append((epoch, iteration))
        self.current_epoch = epoch
        
        # Update plots with epoch boundary
        if self.is_initialized:
            self._update_plots()
            self._update_progress_bars()
            self._update_time_info()
    
    def _update_plots(self):
        """Update all plots with current data"""
        # Update the training loss line with moving average
        self.train_line.set_data(self.iterations, self.train_losses_ma)
        
        # Update the validation loss line with original values (no moving average)
        self.val_line.set_data(self.val_iterations, self.val_losses)
        
        # Update full validation loss line
        self.full_val_line.set_data(self.full_val_iterations, self.full_val_losses)
        
        # Update metrics lines if we have validation data
        if self.val_iterations and 'accuracy' in self.metrics_history and self.metrics_history['accuracy']:
            # Use original accuracy metrics (no moving average)
            self.acc_line.set_data(self.val_iterations, self.metrics_history['accuracy'])
            
            # For composite metrics (precision, recall, f1), use the average 
            if 'f1' in self.metrics_history and self.metrics_history['f1']:
                avg_f1 = [np.mean([self.metrics_history['f1'][cls_idx][i] 
                                 for cls_idx in self.metrics_history['f1']
                                 if i < len(self.metrics_history['f1'][cls_idx])])
                       for i in range(len(self.val_iterations))]
                self.f1_line.set_data(self.val_iterations, avg_f1)
        
        # Update full validation metrics lines
        if self.full_val_iterations and 'accuracy' in self.full_val_metrics_history and self.full_val_metrics_history['accuracy']:
            self.full_acc_line.set_data(self.full_val_iterations, self.full_val_metrics_history['accuracy'])
            
            if 'f1' in self.full_val_metrics_history and self.full_val_metrics_history['f1']:
                avg_f1 = [np.mean([self.full_val_metrics_history['f1'][cls_idx][i] 
                                 for cls_idx in self.full_val_metrics_history['f1']
                                 if i < len(self.full_val_metrics_history['f1'][cls_idx])])
                       for i in range(len(self.full_val_iterations))]
                self.full_f1_line.set_data(self.full_val_iterations, avg_f1)
        
        # Adjust axes for loss plot
        if self.iterations:
            # X-axis
            self.ax_loss.set_xlim(0, max(self.iterations) * 1.1)
            
            # Y-axis
            if not self.max_loss:
                y_max = max(
                    max(self.train_losses_ma) if self.train_losses_ma else 0,
                    max(self.val_losses) if self.val_losses else 0,
                    max(self.full_val_losses) if self.full_val_losses else 0
                )
                self.ax_loss.set_ylim(0, y_max * 1.1)
        
        # Adjust axes for metrics plot
        if self.val_iterations:
            self.ax_metrics.set_xlim(0, max(self.val_iterations) * 1.1)
        
        # Add markers for epoch boundaries
        for ax in [self.ax_loss, self.ax_metrics]:
            # Clear previous epoch markers
            for line in ax.lines[:]:
                if hasattr(line, '_epoch_marker'):
                    line.remove()
            
            # Add new epoch markers
            for epoch, iteration in self.epoch_boundaries:
                line = ax.axvline(x=iteration, color='g', linestyle='--', alpha=0.3)
                line._epoch_marker = True  # Mark as epoch marker
                
                # Add epoch number text (only in loss plot for clarity)
                if ax == self.ax_loss:
                    # Position the text at the bottom of the plot
                    y_min = ax.get_ylim()[0]
                    text_y = y_min + (ax.get_ylim()[1] - y_min) * 0.02  # 2% above bottom
                    ax.text(iteration, text_y, f'E{epoch}', 
                           verticalalignment='bottom', horizontalalignment='center',
                           color='green', fontsize=9)
        
        # Save the updated plot
        plot_filename = 'temp_training_plot.png'
        self.fig.savefig(plot_filename, dpi=100, bbox_inches='tight')
        
        # Update the plot in the dashboard
        self._update_plot_in_dashboard(plot_filename)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import gc
import time
from datetime import datetime
from tqdm.auto import tqdm
from IPython.display import display, HTML, clear_output


import logging
import os
from datetime import datetime

class VideoClassifier:
    """
    An improved class to handle the training, validation, and testing of video classification models
    with dynamic visualization capabilities and detailed logging
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
        validation_freq=5,   # Number of validation runs per epoch
        viz_update_freq=20   # Update visualization every N mini-batches
    ):
        """
        Initialize the video classifier
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            base_model: Base CNN architecture ('resnet18', 'resnet50', etc.)
            temporal_mode: Temporal aggregation method ('attention', 'lstm', 'gru', 'pooling', 'convolution')
            num_classes: Number of classes (3 for Normal, Near Collision, Collision)
            batch_size: Training batch size
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            save_dir: Directory to save checkpoints and results
            experiment_name: Name for this training run (for saving results)
            device: Device to train on (will use CUDA if available if None)
            num_workers: Number of workers for data loading
            class_weights: Optional tensor of class weights for handling imbalanced data
            use_dynamic_viz: Whether to use dynamic visualization (defaults to True)
            validation_freq: Number of validation runs per epoch
            viz_update_freq: Update visualization every N mini-batches
        """
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Save hyperparameters
        self.base_model = base_model
        self.temporal_mode = temporal_mode
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.use_dynamic_viz = use_dynamic_viz
        self.validation_freq = validation_freq
        self.viz_update_freq = viz_update_freq
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{base_model}_{temporal_mode}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Create save directory
        self.save_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING VIDEO CLASSIFIER")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment Name: {self.experiment_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Base Model: {base_model}")
        self.logger.info(f"Temporal Mode: {temporal_mode}")
        self.logger.info(f"Number of Classes: {num_classes}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Learning Rate: {learning_rate}")
        self.logger.info(f"Weight Decay: {weight_decay}")
        self.logger.info(f"Validation Frequency: {validation_freq} times per epoch")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Data Loaders Created:")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Validation batches: {len(self.val_loader)}")
        self.logger.info(f"  Test batches: {len(self.test_loader)}")
        
        # Initialize the model
        self.logger.info("Creating model...")
        self.model = self._create_model()
        self.model.to(self.device)
        self.logger.info("Model created and moved to device")
        
        # Set up class weights for handling imbalanced data
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
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
        
        self.logger.info("VideoClassifier initialization completed successfully")
        self.logger.info("=" * 80)
    
    def _setup_logging(self):
        """Setup detailed logging for training process"""
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
    
    def _create_model(self):
        """Create the model with the correct output layer for multi-class classification"""
        from nexar_arch import EnhancedFrameCNN
        
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
            
            # Replace the final layer with one that outputs the correct number of classes
            model.classifier[-1] = nn.Linear(feature_dim, self.num_classes)
            self.logger.info(f"Replaced final layer: {feature_dim} -> {self.num_classes} classes")
        
        return model
    
    def _setup_training(self):
        """Set up loss function, optimizer, and scheduler"""
        # Loss function depends on number of classes
        if self.num_classes == 2:
            # Binary classification with class weights if provided
            if self.class_weights is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                self.logger.info("Using BCEWithLogitsLoss with class weights")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                self.logger.info("Using BCEWithLogitsLoss")
        else:
            # Multi-class classification with class weights if provided
            if self.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                self.logger.info("Using CrossEntropyLoss with class weights")
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.logger.info("Using CrossEntropyLoss")
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.logger.info(f"Optimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        
        # Learning rate scheduler - Cosine annealing works well for deep learning
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=30,  # Maximum number of iterations
            eta_min=self.learning_rate/100  # Minimum learning rate
        )
        self.logger.info(f"Scheduler: CosineAnnealingLR (T_max=30, eta_min={self.learning_rate/100})")
    
    def _init_visualizer(self, epochs):
        """Initialize the dynamic visualizer with proper parameters"""
        if not self.use_dynamic_viz:
            self.logger.info("Dynamic visualization disabled")
            return
        
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
        self.logger.info("Dynamic visualizer initialized successfully")
    
    def train(self, epochs=30, patience=5, mixed_precision=True):
        """
        Train the model
        
        Args:
            epochs: Maximum number of training epochs
            patience: Early stopping patience (stops if validation loss doesn't improve for this many epochs)
            mixed_precision: Whether to use mixed precision training
        
        Returns:
            history: Dictionary with training history
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"Max Epochs: {epochs}")
        self.logger.info(f"Early Stopping Patience: {patience}")
        self.logger.info(f"Mixed Precision: {mixed_precision}")
        
        # Set up mixed precision training if requested and available
        if mixed_precision and torch.cuda.is_available():
            scaler = torch.amp.GradScaler()
            self.logger.info("Mixed precision training enabled with GradScaler")
        else:
            scaler = None
            self.logger.info("Mixed precision training disabled")
        
        # Initialize visualizer if needed
        if self.use_dynamic_viz:
            self._init_visualizer(epochs)
        
        # For early stopping
        patience_counter = 0
        self.logger.info(f"Initial best validation loss: {self.best_val_loss}")
        
        # Main training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            self.logger.info("=" * 60)
            self.logger.info(f"EPOCH {epoch+1}/{epochs}")
            self.logger.info("=" * 60)
            
            # Update visualizer for new epoch
            if self.visualizer:
                self.visualizer.start_epoch(epoch + 1)
            
            # Train one epoch
            self.logger.info("Starting training phase...")
            train_loss = self._train_epoch(scaler)
            self.logger.info(f"Training phase completed. Average train loss: {train_loss:.6f}")
            
            # Full validation at end of epoch
            self.logger.info("Starting full validation...")
            val_metrics = self._validate()
            self.logger.info("Full validation completed")
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
                    self.logger.info(f"  {class_name}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            
            if 'auc' in val_metrics:
                self.logger.info(f"  AUC: {val_metrics['auc']:.4f}")
            
            # Update visualizer with full validation results
            if self.visualizer:
                current_iteration = (epoch + 1) * len(self.train_loader)
                self.visualizer.update_full_val_metrics(current_iteration, val_metrics['loss'], val_metrics)
            
            # Save this epoch's model and metrics
            self.logger.info("Saving epoch checkpoint...")
            self._save_epoch_checkpoint(epoch + 1, val_metrics)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate updated: {old_lr:.8f} -> {new_lr:.8f}")
            
            # Update history
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
            
            # Check for best model
            self.logger.info("Checking for best model...")
            self.logger.info(f"Current validation loss: {val_metrics['loss']:.6f}")
            self.logger.info(f"Best validation loss so far: {self.best_val_loss:.6f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                improvement = self.best_val_loss - val_metrics['loss']
                self.best_val_loss = val_metrics['loss']
                self.best_val_metrics = val_metrics
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint('best_model.pth')
                self.logger.info(f"*** NEW BEST MODEL FOUND ***")
                self.logger.info(f"Improvement: {improvement:.6f}")
                self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
                self.logger.info(f"Best epoch updated to: {self.best_epoch}")
                self.logger.info("Best model saved")
            else:
                patience_counter += 1
                self.logger.info(f"No improvement. Patience counter: {patience_counter}/{patience}")
                
            # Print progress
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch Summary:")
            self.logger.info(f"  Time: {epoch_time:.2f}s")
            self.logger.info(f"  Train Loss: {train_loss:.6f}")
            self.logger.info(f"  Val Loss: {val_metrics['loss']:.6f}")
            self.logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            if isinstance(val_metrics['f1'], dict):
                avg_f1 = np.mean(list(val_metrics['f1'].values()))
                self.logger.info(f"  Val F1 (avg): {avg_f1:.4f}")
            
            # Mark end of epoch in visualizer
            if self.visualizer:
                current_iteration = (epoch + 1) * len(self.train_loader)
                self.visualizer.mark_epoch(epoch + 1, current_iteration)
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info("=" * 60)
                self.logger.info(f"EARLY STOPPING TRIGGERED")
                self.logger.info(f"No improvement for {patience} epochs")
                self.logger.info(f"Training stopped after {epoch+1} epochs")
                self.logger.info("=" * 60)
                break
                
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
        
        return self.history
    
    def _train_epoch(self, scaler=None):
        """Train for one epoch with dynamic visualization"""
        # Make sure model is in training mode
        self.model.train()
        # Make sure all modules are in training mode
        for module in self.model.modules():
            module.train()
        
        self.logger.debug("Model set to training mode")
        
        train_loss = 0.0
        
        # Training progress bar
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        
        # Calculate validation frequency
        val_step = max(1, len(self.train_loader) // self.validation_freq)
        self.logger.info(f"Will run mini-validation every {val_step} batches")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Log detailed batch info every 50 batches
            if batch_idx % 50 == 0:
                self.logger.debug(f"Processing batch {batch_idx}/{len(self.train_loader)}")
            
            # Get data
            frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
            targets = batch['target']
            
            # Convert string targets to numeric if needed
            if isinstance(targets[0], str):
                # Map class names to indices
                class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
            else:
                targets = targets.to(self.device)
            
            # Make sure model is in training mode before forward pass
            self.model.train()
            
            # Forward pass with mixed precision if available
            self.optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, targets)
                
                try:
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                except RuntimeError as e:
                    if "cudnn RNN backward can only be called in training mode" in str(e):
                        self.logger.warning("Fixing training mode error...")
                        # Re-ensure model is in training mode
                        self.model.train()
                        for module in self.model.modules():
                            module.train()
                        
                        # Try again
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        raise e
            else:
                # Standard training
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            current_loss = loss.item()
            train_loss += current_loss
            avg_loss = train_loss / (batch_idx + 1)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # Update dynamic visualization if enabled
            if self.use_dynamic_viz and self.visualizer:
                # Calculate current iteration
                current_iteration = self.current_epoch * len(self.train_loader) + batch_idx + 1
                
                # Update training loss in visualizer
                self.visualizer.update_train_loss(current_iteration, current_loss)
                
                # Periodically perform validation and update metrics
                if (batch_idx + 1) % val_step == 0 or batch_idx == len(self.train_loader) - 1:
                    self.logger.info(f"Running mini-validation at batch {batch_idx+1}/{len(self.train_loader)}")
                    
                    # Run mini-validation with fewer batches for speed
                    val_metrics = self._mini_validate()
                    
                    self.logger.info(f"Mini-validation results:")
                    self.logger.info(f"  Loss: {val_metrics['loss']:.6f}")
                    self.logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                    self.logger.info(f"  Based on {min(5, len(self.val_loader))} batches")
                    
                    # Update visualization with validation results
                    self.visualizer.update_val_metrics(current_iteration, val_metrics['loss'], val_metrics)
                    
                    # Check if this is the best mini-validation loss
                    self.logger.info(f"Comparing mini-validation loss:")
                    self.logger.info(f"  Current: {val_metrics['loss']:.6f}")
                    self.logger.info(f"  Best so far: {self.best_mini_val_loss:.6f}")
                    
                    if val_metrics['loss'] < self.best_mini_val_loss:
                        improvement = self.best_mini_val_loss - val_metrics['loss']
                        self.best_mini_val_loss = val_metrics['loss']
                        
                        self.logger.info(f"*** NEW BEST MINI-VALIDATION LOSS ***")
                        self.logger.info(f"Improvement: {improvement:.6f}")
                        self.logger.info(f"New best mini-validation loss: {self.best_mini_val_loss:.6f}")
                        
                        # Run full validation
                        self.logger.info("Running full validation due to mini-validation improvement...")
                        full_val_metrics = self._validate()
                        
                        self.logger.info(f"Full validation results:")
                        self.logger.info(f"  Loss: {full_val_metrics['loss']:.6f}")
                        self.logger.info(f"  Accuracy: {full_val_metrics['accuracy']:.4f}")
                        
                        # Update visualizer with full validation results
                        self.visualizer.update_full_val_metrics(current_iteration, full_val_metrics['loss'], full_val_metrics)
                        
                        # Check if this is the best full validation result
                        self.logger.info(f"Comparing full validation loss:")
                        self.logger.info(f"  Current: {full_val_metrics['loss']:.6f}")
                        self.logger.info(f"  Best so far: {self.best_val_loss:.6f}")
                        
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
                            self.logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
                            self.logger.info("Best model saved")
                        else:
                            self.logger.info("Full validation did not improve best loss")
                    else:
                        self.logger.info("Mini-validation did not improve")
                    
                    # IMPORTANT: Set model back to training mode after validation
                    self.model.train()
                    for module in self.model.modules():
                        module.train()
                    self.logger.debug("Model reset to training mode after validation")
            
            # Free memory
            del frames, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average loss
        train_loss /= len(self.train_loader)
        
        return train_loss
    
    def _mini_validate(self, max_batches=25):
        """
        Run a smaller validation for dynamic updates during training with random batch sampling
        
        Args:
            max_batches: Maximum number of batches to use for quick validation
        """
        self.logger.debug(f"Starting mini-validation with max {max_batches} batches (shuffled)")
        
        # Create a shuffled DataLoader for mini-validation if it doesn't exist
        if not hasattr(self, '_mini_val_loader'):
            self.logger.debug("Creating shuffled mini-validation DataLoader")
            self._mini_val_loader = DataLoader(
                self.val_loader.dataset, 
                batch_size=self.val_loader.batch_size,
                shuffle=True,  # Random sampling each time
                num_workers=self.val_loader.num_workers,
                pin_memory=self.val_loader.pin_memory,
                drop_last=False
            )
            self.logger.debug(f"Mini-validation DataLoader created with {len(self._mini_val_loader)} total batches")
        
        # Switch to eval mode
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        batches_processed = 0
        total_samples = 0
        
        with torch.no_grad():
            # Use the shuffled loader - each iteration will give different batches
            for batch_idx, batch in enumerate(self._mini_val_loader):
                if batch_idx >= max_batches:
                    break
                    
                # Get data
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)
                targets = batch['target']
                
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
                val_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                batches_processed += 1
                total_samples += len(targets)
                
                # Log every 10 batches for debugging
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Mini-val batch {batch_idx}: loss={loss.item():.6f}, samples={len(targets)}")
                
                # Free memory
                del frames, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate metrics on the subset
        val_loss /= batches_processed
        
        self.logger.debug(f"Mini-validation processed {batches_processed} batches ({total_samples} samples)")
        self.logger.debug(f"Mini-validation average loss: {val_loss:.6f}")
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        self.logger.debug(f"Mini-validation final tensor shapes: outputs={all_outputs.shape}, targets={all_targets.shape}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = val_loss
        metrics['batches_used'] = batches_processed
        metrics['samples_used'] = total_samples
        
        # Add some statistics for comparison with full validation
        metrics['percentage_of_val_set'] = (total_samples / len(self.val_loader.dataset)) * 100
        
        self.logger.debug(f"Mini-validation metrics calculated:")
        self.logger.debug(f"  Samples: {total_samples}/{len(self.val_loader.dataset)} ({metrics['percentage_of_val_set']:.1f}%)")
        self.logger.debug(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.debug(f"  Loss: {metrics['loss']:.6f}")
        
        # Note: Model will be set back to training mode in the calling function
        
        return metrics
    
    def _validate(self):
        """Full validation on the entire validation set"""
        self.logger.debug("Starting full validation")
        
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        batches_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Get data
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
                targets = batch['target']
                
                # Convert string targets to numeric if needed
                if isinstance(targets[0], str):
                    # Map class names to indices
                    class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)
                
                # Store results
                val_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                batches_processed += 1
                
                # Free memory
                del frames, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        val_loss /= batches_processed
        metrics = self._calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = val_loss
        metrics['batches_used'] = batches_processed
        
        self.logger.debug(f"Full validation processed {batches_processed} batches")
        self.logger.debug(f"Full validation average loss: {val_loss:.6f}")
        
        return metrics
    
    def _calculate_metrics(self, outputs, targets):
        """Calculate classification metrics including per-class metrics"""
        self.logger.debug(f"Calculating metrics for {len(targets)} samples")
        
        # Get predictions
        if self.num_classes == 2:
            # Binary classification
            y_pred = (torch.sigmoid(outputs) > 0.5).float()
            y_prob = torch.sigmoid(outputs).numpy()
        else:
            # Multi-class classification
            y_pred = torch.argmax(outputs, dim=1)
            y_prob = torch.softmax(outputs, dim=1).numpy()
        
        # Convert to numpy for sklearn metrics
        y_true = targets.numpy()
        y_pred = y_pred.numpy()
        
        # Calculate overall metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Calculate per-class precision, recall, and F1
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        
        for cls in range(self.num_classes):
            # For each class, treat it as a binary classification problem
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))
            
            # Calculate precision
            if true_positives + false_positives > 0:
                precision_per_class[cls] = true_positives / (true_positives + false_positives)
            else:
                precision_per_class[cls] = 0.0
            
            # Calculate recall
            if true_positives + false_negatives > 0:
                recall_per_class[cls] = true_positives / (true_positives + false_negatives)
            else:
                recall_per_class[cls] = 0.0
            
            # Calculate F1 score
            if precision_per_class[cls] + recall_per_class[cls] > 0:
                f1_per_class[cls] = 2 * (precision_per_class[cls] * recall_per_class[cls]) / (precision_per_class[cls] + recall_per_class[cls])
            else:
                f1_per_class[cls] = 0.0
        
        # Add per-class metrics to the result
        metrics['precision'] = precision_per_class
        metrics['recall'] = recall_per_class
        metrics['f1'] = f1_per_class
        
        # Add AUC for multi-class (one-vs-rest)
        try:
            if self.num_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:
                # One-vs-rest AUC
                metrics['auc'] = roc_auc_score(
                    np.eye(self.num_classes)[y_true],  # Convert to one-hot
                    y_prob, 
                    multi_class='ovr',
                    average='weighted'
                )
        except Exception as e:
            self.logger.warning(f"Error calculating AUC: {e}")
            metrics['auc'] = 0.5
            
        return metrics
    
    def _save_epoch_checkpoint(self, epoch, val_metrics):
        """Save checkpoint for each epoch"""
        checkpoint_name = f'checkpoint_epoch{epoch}.pth'
        self._save_checkpoint(checkpoint_name)
        self._save_validation_results(f'validation_epoch{epoch}', val_metrics)
        self.logger.debug(f"Saved checkpoint: {checkpoint_name}")
    
    def _save_validation_results(self, filename, metrics):
        """Save validation results to JSON file"""
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
        
        self.logger.debug(f"Saved validation results: {filename}.json")
    
    def test(self):
        """Test the model on the test set"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING TESTING")
        self.logger.info("=" * 80)
        
        self.model.eval()
        test_loss = 0.0
        all_targets = []
        all_outputs = []
        all_ids = []
        
        batches_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Get data
                frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(self.device)  # [B, C, T, H, W]
                targets = batch['target']
                ids = batch['id'] if 'id' in batch else [f"sample_{i}" for i in range(len(targets))]
                
                # Convert string targets to numeric if needed
                if isinstance(targets[0], str):
                    # Map class names to indices
                    class_map = {'Normal': 0, 'Near Collision': 1, 'Collision': 2}
                    targets = torch.tensor([class_map[t] for t in targets]).to(self.device)
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, targets)
                
                # Store results
                test_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_ids.extend(ids)
                batches_processed += 1
                
                # Free memory
                del frames, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate results
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        test_loss /= batches_processed
        metrics = self._calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = test_loss
        
        # Save test results
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
        
        return metrics
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
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
            }
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        self.logger.debug(f"Saved checkpoint: {filename}")
    
    def _load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metrics = checkpoint['best_val_metrics']
        self.best_epoch = checkpoint['best_epoch']
        self.logger.info(f"Loaded model from epoch {self.best_epoch} with validation loss {self.best_val_loss:.6f}")
    
    def _save_history(self):
        """Save training history to CSV"""
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        self.logger.info("Training history saved to CSV")
    
    def _plot_training_history(self):
        """Plot training history graphs at the end of training"""
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
        """Plot confusion matrix"""
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
        """Save model predictions to CSV"""
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
        
        Args:
            num_samples: Number of samples to visualize
        """
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
        """
        # Check if model uses attention
        if self.temporal_mode != 'attention' or not hasattr(self.model, 'temporal_aggregation') or not hasattr(self.model.temporal_aggregation, 'get_attention_weights'):
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
            attn_weights = self.model.temporal_aggregation.last_attn_weights
        
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

# Example for using the Enhanced VideoClassifier

def compute_class_weights(dataset):
    """
    Compute class weights based on the distribution of classes in the dataset
    
    Args:
        dataset: Dataset with 'video_type' field in metadata_df
        
    Returns:
        torch.Tensor with weights for each class
    """
    if not hasattr(dataset, 'metadata_df') or 'video_type' not in dataset.metadata_df.columns:
        print("Dataset does not have video_type labels. Using equal class weights.")
        return None
    
    # Count class occurrences
    class_counts = dataset.metadata_df['video_type'].value_counts().to_dict()
    print(f"Class distribution: {class_counts}")
    
    # Ensure we have counts for all classes
    all_classes = ['Normal', 'Near Collision', 'Collision']
    for class_name in all_classes:
        if class_name not in class_counts:
            class_counts[class_name] = 0
    
    # Convert to sorted list of counts (Normal, Near Collision, Collision)
    counts = [class_counts[c] for c in all_classes]
    print(f"Counts in order [Normal, Near Collision, Collision]: {counts}")
    
    # Compute weights as inverse of frequency
    total_samples = sum(counts)
    class_weights = [total_samples / (len(counts) * c) if c > 0 else 1.0 for c in counts]
    
    print(f"Computed class weights [Normal, Near Collision, Collision]: {class_weights}")
    
    # Convert to tensor
    return torch.tensor(class_weights, dtype=torch.float32)


def run_experiment(
    train_data, 
    val_data, 
    test_data,
    base_model='resnet18',
    temporal_mode='attention',
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=8,
    epochs=30,
    experiment_name=None,
    use_class_weights=True,
    save_dir='model_experiments',
    use_dynamic_viz=True,
    validation_freq=5,   # Number of validation runs per epoch
    viz_update_freq=20   # Update visualization every N mini-batches
):
    """
    Run a video classification experiment with dynamic visualization
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        base_model: Base CNN architecture
        temporal_mode: Temporal aggregation method
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        batch_size: Batch size
        epochs: Maximum number of epochs
        experiment_name: Name for this experiment (for saving results)
        use_class_weights: Whether to use class weights for handling imbalanced data
        save_dir: Directory to save results
        use_dynamic_viz: Whether to use dynamic visualization
        validation_freq: Number of validation runs per epoch
        viz_update_freq: Update visualization every N mini-batches
        
    Returns:
        classifier: Trained VideoClassifier object
    """
    # Set experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{base_model}_{temporal_mode}_{timestamp}"
    
    # Calculate class weights if requested
    class_weights = compute_class_weights(train_data) if use_class_weights else None
    if class_weights is not None:
        print(f"Using class weights: {class_weights}")
    
    # Create classifier
    classifier = VideoClassifier(
        train_dataset=train_data,
        val_dataset=val_data,
        test_dataset=test_data,
        base_model=base_model,
        temporal_mode=temporal_mode,
        num_classes=3,  # Normal, Near Collision, Collision
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_dir=save_dir,
        experiment_name=experiment_name,
        class_weights=class_weights,
        use_dynamic_viz=use_dynamic_viz,
        validation_freq=validation_freq,
        viz_update_freq=viz_update_freq
    )
    
    # Train the model
    history = classifier.train(epochs=epochs)
    
    # Test the model
    test_metrics = classifier.test()
    
    # Return the trained classifier
    return classifier


def run_grid_search(
    train_data, 
    val_data, 
    test_data,
    base_models=['resnet18', 'mobilenet_v3_small', 'convnext_tiny'],
    temporal_modes=['attention', 'lstm', 'pooling'],
    learning_rates=[1e-4],
    batch_sizes=[8],
    epochs=20,
    save_dir='grid_search_results'
):
    """
    Run a grid search over multiple model configurations
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        base_models: List of base models to try
        temporal_modes: List of temporal modes to try
        learning_rates: List of learning rates to try
        batch_sizes: List of batch sizes to try
        epochs: Maximum number of epochs for each experiment
        save_dir: Directory to save results
    
    Returns:
        results_df: DataFrame with experiment results
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for this grid search
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_search_dir = os.path.join(save_dir, f"grid_search_{timestamp}")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Store results
    results = []
    
    # Run all combinations
    for base_model in base_models:
        for temporal_mode in temporal_modes:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    # Create experiment name
                    experiment_name = f"{base_model}_{temporal_mode}_lr{lr}_bs{batch_size}"
                    print(f"\n\n{'='*80}")
                    print(f"Starting experiment: {experiment_name}")
                    print(f"{'='*80}\n")
                    
                    try:
                        # Run experiment
                        start_time = time.time()
                        
                        classifier = run_experiment(
                            train_data=train_data,
                            val_data=val_data,
                            test_data=test_data,
                            base_model=base_model,
                            temporal_mode=temporal_mode,
                            learning_rate=lr,
                            batch_size=batch_size,
                            epochs=epochs,
                            experiment_name=experiment_name,
                            save_dir=grid_search_dir
                        )
                        
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Get best validation metrics
                        val_metrics = classifier.best_val_metrics
                        
                        # Get test metrics
                        test_metrics = classifier.test()
                        
                        # Add to results
                        results.append({
                            'base_model': base_model,
                            'temporal_mode': temporal_mode,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'best_epoch': classifier.best_epoch,
                            'training_duration': duration,
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy'],
                            'val_f1': val_metrics['f1'],
                            'val_auc': val_metrics.get('auc', 0),
                            'test_loss': test_metrics['loss'],
                            'test_accuracy': test_metrics['accuracy'],
                            'test_f1': test_metrics['f1'],
                            'test_auc': test_metrics.get('auc', 0),
                        })
                        
                        # Save intermediate results
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(os.path.join(grid_search_dir, 'grid_search_results.csv'), index=False)
                        
                    except Exception as e:
                        print(f"Error in experiment {experiment_name}: {str(e)}")
                        results.append({
                            'base_model': base_model,
                            'temporal_mode': temporal_mode,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'error': str(e)
                        })
                        
                        # Save error log
                        with open(os.path.join(grid_search_dir, f"{experiment_name}_error.log"), 'w') as f:
                            f.write(f"Error in experiment {experiment_name}: {str(e)}")
    
    # Create final results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    results_df.to_csv(os.path.join(grid_search_dir, 'final_grid_search_results.csv'), index=False)
    
    # Plot grid search results
    _plot_grid_search_results(results_df, grid_search_dir)
    
    return results_df


def _plot_grid_search_results(results_df, save_dir):
    """
    Plot grid search results
    
    Args:
        results_df: DataFrame with grid search results
        save_dir: Directory to save plots
    """
    # Check if results DataFrame has required columns
    required_cols = ['base_model', 'temporal_mode', 'test_accuracy', 'test_f1']
    if not all(col in results_df.columns for col in required_cols):
        print("Results DataFrame doesn't have required columns for plotting")
        return
    
    # Filter out rows with errors
    if 'error' in results_df.columns:
        results_df = results_df[results_df['error'].isna()]
    
    if len(results_df) == 0:
        print("No successful experiments to plot")
        return
    
    # Plot by base model and temporal mode
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar chart
    # Group by base_model and temporal_mode, then plot test accuracy and F1
    grouped_results = results_df.groupby(['base_model', 'temporal_mode']).agg({
        'test_accuracy': 'mean',
        'test_f1': 'mean'
    }).reset_index()
    
    # Create x positions for bars
    base_models = grouped_results['base_model'].unique()
    temporal_modes = grouped_results['temporal_mode'].unique()
    x = np.arange(len(base_models))
    width = 0.15  # width of bars
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot accuracy
    for i, mode in enumerate(temporal_modes):
        mode_data = grouped_results[grouped_results['temporal_mode'] == mode]
        mode_data = mode_data.set_index('base_model').reindex(base_models).reset_index()
        ax1.bar(x + (i - len(temporal_modes)/2 + 0.5) * width, 
                mode_data['test_accuracy'], 
                width, 
                label=f'{mode}')
    
    ax1.set_title('Test Accuracy by Model and Temporal Mode')
    ax1.set_xticks(x)
    ax1.set_xticklabels(base_models, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot F1 score
    for i, mode in enumerate(temporal_modes):
        mode_data = grouped_results[grouped_results['temporal_mode'] == mode]
        mode_data = mode_data.set_index('base_model').reindex(base_models).reset_index()
        ax2.bar(x + (i - len(temporal_modes)/2 + 0.5) * width, 
                mode_data['test_f1'], 
                width, 
                label=f'{mode}')
    
    ax2.set_title('Test F1 Score by Model and Temporal Mode')
    ax2.set_xticks(x)
    ax2.set_xticklabels(base_models, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_search_results.png'))
    plt.close()
    
    # Create a heatmap for test accuracy
    plt.figure(figsize=(10, 8))
    pivot_acc = pd.pivot_table(
        grouped_results, 
        values='test_accuracy', 
        index='base_model', 
        columns='temporal_mode'
    )
    sns.heatmap(pivot_acc, annot=True, cmap='YlGnBu', fmt='.3f', vmin=0.5, vmax=1.0)
    plt.title('Test Accuracy Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_heatmap.png'))
    plt.close()
    
    # Create a heatmap for test F1 score
    plt.figure(figsize=(10, 8))
    pivot_f1 = pd.pivot_table(
        grouped_results, 
        values='test_f1', 
        index='base_model', 
        columns='temporal_mode'
    )
    sns.heatmap(pivot_f1, annot=True, cmap='YlGnBu', fmt='.3f', vmin=0.5, vmax=1.0)
    plt.title('Test F1 Score Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_heatmap.png'))
    plt.close()


def visualize_model_predictions(model, dataset, num_samples=5, save_dir=None):
    """
    Visualize model predictions on random samples from the dataset
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    # Create DataLoader with batch size 1
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
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
            frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(device)  # [B, C, T, H, W]
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
            outputs = model(frames)
            
            # Get predictions
            if model.num_classes == 2:
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
            # Convert from [B, C, T, H, W] back to [T, H, W, C] for display
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
    
    # Save or show the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prediction_visualization.png'))
        plt.close()
    else:
        plt.show()


def visualize_attention_weights(model, dataset, save_dir=None):
    """
    Visualize attention weights for a model with attention-based temporal aggregation
    
    Args:
        model: Trained model with temporal_mode='attention'
        dataset: Dataset to sample from
        save_dir: Directory to save visualizations
    """
    # Check if model uses attention
    if not hasattr(model, 'temporal_aggregation') or not hasattr(model.temporal_aggregation, 'get_attention_weights'):
        print("Model does not support attention visualization")
        return
    
    # Create DataLoader with batch size 1
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a sample
    batch = next(iter(loader))
    
    # Get data
    frames = batch['frames'].permute(0, 4, 1, 2, 3).float().to(device)  # [B, C, T, H, W]
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
        _ = model(frames)
        attn_weights = model.temporal_aggregation.last_attn_weights
    
    if attn_weights is None:
        print("No attention weights available")
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
    a
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
        im = axes[1, i].imshow([[attn_scores[i, i]]], cmap='hot', vmin=0, vmax=attn_scores.max())
        axes[1, i].set_title(f"Attn: {attn_scores[i, i]:.2f}")
        axes[1, i].axis('off')
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Add title
    class_map_rev = {0: 'Normal', 1: 'Near Collision', 2: 'Collision'}
    fig.suptitle(f"Attention Visualization for {class_map_rev[target_idx]} Sample", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save or show the figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'attention_visualization.png'))
        plt.close()
    else:
        plt.show()