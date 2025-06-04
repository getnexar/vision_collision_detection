import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Try to import IPython/Jupyter specific modules
try:
    from IPython.display import display, HTML, Javascript, clear_output
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# Try to import distributed modules
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class DynamicTrainingVisualizer:
    """
    An improved class to visualize training metrics dynamically during training with
    persistent display and efficient updates. Shows moving average for smoother visualization.
    Now includes full validation results tracking and distributed training support.
    
    In distributed training, only the master process (rank 0) will display visualizations.
    Other processes will have dummy methods that do nothing.
    """
    def __init__(self, max_loss=None, num_epochs=None, num_iterations_per_epoch=None, 
                 update_freq=20, num_classes=3, class_names=None, moving_avg_window=29,
                 is_master_process=True, enable_jupyter_display=None):
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
            is_master_process: Whether this is the master process (for distributed training)
            enable_jupyter_display: Whether to enable Jupyter display (auto-detect if None)
        """
        # Auto-detect if we're running in a Jupyter environment and if this is master process
        if enable_jupyter_display is None:
            enable_jupyter_display = JUPYTER_AVAILABLE
        
        # Only enable visualization on master process and in Jupyter environment
        self.enabled = is_master_process and enable_jupyter_display
        self.is_master_process = is_master_process
        
        if not self.enabled:
            # Create dummy visualizer for non-master processes or non-Jupyter environments
            self._init_dummy()
            return
        
        # Normal initialization for master process in Jupyter
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
        
        print(f"DynamicTrainingVisualizer initialized (Master Process: {self.is_master_process}, Enabled: {self.enabled})")
    
    def _init_dummy(self):
        """Initialize dummy visualizer for non-master processes"""
        self.enabled = False
        self.is_master_process = False
        
        # Create dummy attributes to prevent attribute errors
        dummy_attrs = [
            'max_loss', 'num_epochs', 'num_iterations_per_epoch', 'update_freq',
            'num_classes', 'class_names', 'moving_avg_window', 'train_losses',
            'train_losses_ma', 'val_losses', 'iterations', 'val_iterations',
            'current_epoch', 'current_iteration', 'epoch_boundaries',
            'full_val_losses', 'full_val_iterations', 'full_val_metrics_history',
            'metrics_history', 'best_metrics', 'start_time', 'last_update_time',
            'epoch_start_time', 'output_html', 'is_initialized', 'animation'
        ]
        
        for attr in dummy_attrs:
            setattr(self, attr, None)
        
        print("DynamicTrainingVisualizer initialized in dummy mode (non-master process or non-Jupyter environment)")
    
    def _calculate_moving_average(self, values, window_size):
        """Calculate moving average with specified window size"""
        if not self.enabled or len(values) < 2:
            return values if self.enabled else []
        
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
        if not self.enabled:
            return
            
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
            
            .distributed-info {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 4px;
                padding: 8px;
                margin-bottom: 10px;
                color: #0c5460;
                font-size: 0.9em;
            }
        </style>
        """
        
        # Display the CSS once
        display(HTML(self.css))
    
    def initialize_display(self):
        """Create the initial display structure with placeholders"""
        if not self.enabled or self.is_initialized:
            return
                
        # Create the main dashboard container
        dashboard_html = """
        <div class="training-dashboard" id="training-dashboard">
            <h2>Training Progress</h2>
            
            <div class="distributed-info" id="distributed-info">
                <strong>Distributed Training Mode:</strong> Visualization running on master process only
            </div>
            
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
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
        # Read the image file and convert to base64
        import base64
        try:
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
        except Exception as e:
            print(f"Error updating plot in dashboard: {e}")
    
    def _create_metrics_table_html(self, latest_metrics=None):
        """Create HTML for the metrics table"""
        if not self.enabled:
            return ""
            
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
        if not self.enabled:
            return
            
        table_html = self._create_metrics_table_html(latest_metrics)
        
        # Use JavaScript to update the table container
        js_code = f"""
        document.getElementById('metrics-table-container').innerHTML = `{table_html}`;
        """
        display(Javascript(js_code))
    
    def _update_progress_bars(self, force_update=False):
        """Update the progress bars in the dashboard"""
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
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
        js_code = f"""
        document.getElementById('elapsed-time').innerText = "Elapsed: {elapsed_str}";
        document.getElementById('eta').innerText = "ETA: {eta_str}";
        document.getElementById('iter-per-sec').innerText = "Iterations/sec: {iter_per_sec:.2f}";
        """
        display(Javascript(js_code))
    
    def start_epoch(self, epoch_num):
        """Mark the start of a new epoch"""
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
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
        if not self.enabled:
            return
            
        self.epoch_boundaries.append((epoch, iteration))
        self.current_epoch = epoch
        
        # Update plots with epoch boundary
        if self.is_initialized:
            self._update_plots()
            self._update_progress_bars()
            self._update_time_info()
    
    def _update_plots(self):
        """Update all plots with current data"""
        if not self.enabled:
            return
            
        try:
            # Update the training loss line with moving average
            if self.iterations and self.train_losses_ma:
                self.train_line.set_data(self.iterations, self.train_losses_ma)
            
            # Update the validation loss line with original values (no moving average)
            if self.val_iterations and self.val_losses:
                self.val_line.set_data(self.val_iterations, self.val_losses)
            
            # Update full validation loss line
            if self.full_val_iterations and self.full_val_losses:
                self.full_val_line.set_data(self.full_val_iterations, self.full_val_losses)
            
            # Update metrics lines if we have validation data
            if self.val_iterations and 'accuracy' in self.metrics_history and self.metrics_history['accuracy']:
                # Use original accuracy metrics (no moving average)
                self.acc_line.set_data(self.val_iterations, self.metrics_history['accuracy'])
                
                # For composite metrics (precision, recall, f1), use the average 
                if 'f1' in self.metrics_history and self.metrics_history['f1']:
                    try:
                        avg_f1 = [np.mean([self.metrics_history['f1'][cls_idx][i] 
                                         for cls_idx in self.metrics_history['f1']
                                         if i < len(self.metrics_history['f1'][cls_idx])])
                               for i in range(len(self.val_iterations))]
                        self.f1_line.set_data(self.val_iterations, avg_f1)
                    except (IndexError, ValueError):
                        # Handle cases where metrics history is inconsistent
                        pass
            
            # Update full validation metrics lines
            if self.full_val_iterations and 'accuracy' in self.full_val_metrics_history and self.full_val_metrics_history['accuracy']:
                self.full_acc_line.set_data(self.full_val_iterations, self.full_val_metrics_history['accuracy'])
                
                if 'f1' in self.full_val_metrics_history and self.full_val_metrics_history['f1']:
                    try:
                        avg_f1 = [np.mean([self.full_val_metrics_history['f1'][cls_idx][i] 
                                         for cls_idx in self.full_val_metrics_history['f1']
                                         if i < len(self.full_val_metrics_history['f1'][cls_idx])])
                               for i in range(len(self.full_val_iterations))]
                        self.full_f1_line.set_data(self.full_val_iterations, avg_f1)
                    except (IndexError, ValueError):
                        # Handle cases where metrics history is inconsistent
                        pass
            
            # Adjust axes for loss plot
            if self.iterations:
                # X-axis
                self.ax_loss.set_xlim(0, max(self.iterations) * 1.1)
                
                # Y-axis
                if not self.max_loss:
                    y_values = []
                    if self.train_losses_ma:
                        y_values.extend(self.train_losses_ma)
                    if self.val_losses:
                        y_values.extend(self.val_losses)
                    if self.full_val_losses:
                        y_values.extend(self.full_val_losses)
                    
                    if y_values:
                        y_max = max(y_values)
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
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    # Dummy methods for non-master processes (these do nothing but prevent errors)
    def dummy_method(self, *args, **kwargs):
        """Dummy method that does nothing for non-master processes"""
        pass
    
    # If this is not the master process, replace all methods with dummy methods
    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        
        # If this is not enabled and it's a method (but not a special method), return dummy
        if (not object.__getattribute__(self, 'enabled') and 
            callable(attr) and 
            not name.startswith('_') and 
            name not in ['dummy_method', 'enabled', 'is_master_process']):
            return object.__getattribute__(self, 'dummy_method')
        
        return attr


# Factory function for creating visualizer with distributed support
def create_distributed_visualizer(num_epochs=None, num_iterations_per_epoch=None, 
                                 num_classes=3, is_master_process=None, **kwargs):
    """
    Factory function to create a DynamicTrainingVisualizer with proper distributed setup
    
    Args:
        num_epochs: Total number of epochs
        num_iterations_per_epoch: Number of iterations per epoch  
        num_classes: Number of classes
        is_master_process: Whether this is the master process (auto-detect if None)
        **kwargs: Other arguments to pass to DynamicTrainingVisualizer
    
    Returns:
        DynamicTrainingVisualizer instance (enabled only on master process)
    """
    # Auto-detect if this is the master process
    if is_master_process is None:
        if DISTRIBUTED_AVAILABLE and dist.is_initialized():
            is_master_process = dist.get_rank() == 0
        else:
            is_master_process = True
    
    # Auto-detect Jupyter environment
    enable_jupyter_display = JUPYTER_AVAILABLE
    
    visualizer = DynamicTrainingVisualizer(
        num_epochs=num_epochs,
        num_iterations_per_epoch=num_iterations_per_epoch,
        num_classes=num_classes,
        is_master_process=is_master_process,
        enable_jupyter_display=enable_jupyter_display,
        **kwargs
    )
    
    return visualizer