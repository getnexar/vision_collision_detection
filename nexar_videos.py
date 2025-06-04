import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import decord
from tqdm.auto import tqdm
import random
import cv2
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import the augmentation code you have
from nexar_video_aug import create_video_transforms

def _find_video_and_sensor_paths(video_id, base_dirs, sensor_subdir):
    for base_dir in base_dirs:
        video_dir = os.path.join(base_dir, video_id)
        if not os.path.exists(video_dir):
            continue

        files = os.listdir(video_dir)
        video_file = None
        
        for file in files:
            if file.endswith('.mp4') or file.endswith('.mov'):
                video_file = file
                break
                
        if video_file:
            video_path = os.path.join(video_dir, video_file)
            sensor_path = os.path.join(video_dir, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
            return video_path, sensor_path if os.path.exists(sensor_path) else None

    return None, None


class NvidiaDashcamDataset(Dataset):
    """Dataset for loading NVIDIA Dashcam video frames with IMU sensor data"""

    def __init__(self, metadata_df, base_dirs, fps=10, duration=5, is_train=True,
                 skip_missing=True, transform=None, sample_strategy='random',
                 sensor_subdir='signals', time_column=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.base_dirs = base_dirs if isinstance(base_dirs, list) else [base_dirs]
        self.fps = fps
        self.duration = duration
        self.is_train = is_train
        self.skip_missing = skip_missing
        self.transform = transform
        self.sample_strategy = sample_strategy
        self.sensor_subdir = sensor_subdir
        self.time_column = time_column

        # Update the valid strategies list to include 'center'
        if self.sample_strategy not in ['random', 'metadata_time', 'center']:
            self.sample_strategy = 'random'

        if self.sample_strategy == 'metadata_time' and (time_column is None or time_column not in self.metadata_df.columns):
            self.sample_strategy = 'random'

        self.class_mapping = {
            'Normal': 'Normal',
            'Near Collision': 'Near Collision',
            'Collision': 'Collision'
        }

        self.video_paths = []
        self.sensor_paths = []
        self.valid_indices = []

        for idx, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df), desc="Checking files"):
            video_id = row['id']
            video_path, sensor_path = _find_video_and_sensor_paths(video_id, self.base_dirs, self.sensor_subdir)

            if video_path:
                self.video_paths.append(video_path)
                self.sensor_paths.append(sensor_path)
                self.valid_indices.append(idx)
            elif not self.skip_missing:
                fallback_video = os.path.join(self.base_dirs[0], video_id, f"{video_id}.mp4")
                fallback_sensor = os.path.join(self.base_dirs[0], video_id, self.sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
                self.video_paths.append(fallback_video)
                self.sensor_paths.append(fallback_sensor)
                self.valid_indices.append(idx)

        if self.skip_missing:
            self.metadata_df = self.metadata_df.iloc[self.valid_indices].reset_index(drop=True)

    def __repr__(self):
        """
        Return a professional-looking summary of the dataset.
        """
        if not hasattr(self, 'metadata_df') or len(self.metadata_df) == 0:
            return f"<{self.__class__.__name__}: Empty dataset>"
        
        # Count examples by class
        class_counts = self.metadata_df['video_type'].value_counts().sort_index()
        total_count = len(self.metadata_df)
        
        # Create a nice-looking string representation
        header = f"🎥 {self.__class__.__name__} Summary"
        separator = "═" * len(header)
        
        # Basic information
        basic_info = [
            f"📊 Total Videos: {total_count}",
            f"🎞️ FPS: {self.fps}",
            f"⏱️ Duration: {self.duration} seconds"
        ]
        
        # Class distribution
        class_info = ["📋 Class Distribution:"]
        max_class_name_len = max([len(str(c)) for c in class_counts.index])
        
        for class_name, count in class_counts.items():
            percentage = (count / total_count) * 100
            bar_length = int(percentage / 5)  # 20 chars = 100%
            bar = "█" * bar_length + "░" * (20 - bar_length)
            class_info.append(f"   {class_name:<{max_class_name_len}} │ {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Valid indices information
        valid_info = []
        if hasattr(self, 'valid_indices') and len(self.valid_indices) != total_count:
            valid_info = [
                f"📁 Valid Videos: {len(self.valid_indices)} / {total_count} ({len(self.valid_indices)/total_count*100:.1f}%)"
            ]
        
        # Additional configuration
        config_info = [
            f"🔧 Training Mode: {'✓' if self.is_train else '✗'}",
            f"🔍 Sampling Strategy: {self.sample_strategy}"
        ]
        
        # Combine all parts with proper spacing
        parts = [header, separator] + basic_info + [""] + class_info
        if valid_info:
            parts += [""] + valid_info
        parts += [""] + config_info
        
        return "\n".join(parts)
    
    # To display the dataset as a beautiful colored table in Jupyter notebooks
    def _repr_html_(self):
        """
        Return an HTML representation for Jupyter notebooks.
        """
        if not hasattr(self, 'metadata_df') or len(self.metadata_df) == 0:
            return f"<h3>{self.__class__.__name__}: Empty dataset</h3>"
        
        # Count examples by class
        class_counts = self.metadata_df['video_type'].value_counts().sort_index()
        total_count = len(self.metadata_df)
        
        # Class colors
        class_colors = {
            'Normal': '#2ca02c',
            'Near Collision': '#ff7f0e',
            'Collision': '#d62728'
        }
        
        # Default color for unknown classes
        default_colors = ['#1f77b4', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_idx = 0
        
        # Generate HTML
        html = f"""
        <style>
        .dataset-summary {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
            margin: 20px 0;
            max-width: 800px;
        }}
        .dataset-header {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #ddd;
        }}
        .dataset-info {{
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }}
        .info-item {{
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .class-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .class-table th {{
            text-align: left;
            padding: 8px;
            border-bottom: 2px solid #ddd;
        }}
        .class-table td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .bar-container {{
            width: 200px;
            background-color: #f1f1f1;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar {{
            height: 20px;
            border-radius: 4px;
        }}
        .config-item {{
            display: inline-block;
            margin-right: 15px;
            margin-top: 10px;
            padding: 3px 8px;
            background-color: #f5f5f5;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        </style>
        
        <div class="dataset-summary">
            <div class="dataset-header">🎥 {self.__class__.__name__} Summary</div>
            
            <div class="dataset-info">
                <div class="info-item"><strong>📊 Total Videos:</strong> {total_count}</div>
                <div class="info-item"><strong>🎞️ FPS:</strong> {self.fps}</div>
                <div class="info-item"><strong>⏱️ Duration:</strong> {self.duration} seconds</div>
        """
        
        # Add valid videos info if available
        if hasattr(self, 'valid_indices') and len(self.valid_indices) != total_count:
            html += f"""
                <div class="info-item"><strong>📁 Valid Videos:</strong> {len(self.valid_indices)} / {total_count} ({len(self.valid_indices)/total_count*100:.1f}%)</div>
            """
        
        html += """
            </div>
            
            <table class="class-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Distribution</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add class rows
        for class_name, count in class_counts.items():
            percentage = (count / total_count) * 100
            
            # Get color for this class
            if class_name in class_colors:
                color = class_colors[class_name]
            else:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1
                
            html += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                        <td>
                            <div class="bar-container">
                                <div class="bar" style="width: {percentage}%; background-color: {color};"></div>
                            </div>
                        </td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <div>
        """
        
        # Add configuration details
        html += f"""
                <div class="config-item">🔧 Training Mode: {'✓' if self.is_train else '✗'}</div>
                <div class="config-item">🔍 Sampling Strategy: {self.sample_strategy}</div>
        """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def __len__(self):
        return len(self.video_paths)
    
    def _load_and_sync_sensor_data(self, video_path, sensor_path):
        """Load sensor data and synchronize with video frames"""
        # Default empty sensor data
        empty_sensor = np.zeros((self.fps * self.duration, 4), dtype=np.float32)
        
        if sensor_path is None or not os.path.exists(sensor_path):
            return empty_sensor
        
        try:
            # Load sensor data
            df_sensor = pd.read_csv(sensor_path, index_col=0)
            
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if frame_count == 0 or fps == 0:
                return empty_sensor
            
            # Calculate video duration
            duration_sec = frame_count / fps
            
            # Convert sensor time to relative time (seconds from start)
            sensor_start_time = df_sensor['time_sec'].iloc[0]
            df_sensor['relative_time_sec'] = df_sensor['time_sec'] - sensor_start_time
            df_sensor = df_sensor.set_index('relative_time_sec')
            
            # Create timestamps for each video frame
            video_times = pd.Series([i / fps for i in range(frame_count)], name='video_time_sec')
            
            # Interpolate sensor data to match video frame times
            aligned_sensor = df_sensor.reindex(df_sensor.index.union(video_times)).interpolate('index').loc[video_times]
            aligned_sensor.reset_index(drop=True, inplace=True)
            aligned_sensor['video_time_sec'] = video_times
            
            # Create synced DataFrame
            synced_df = aligned_sensor[['video_time_sec', 'accel_x_G', 'accel_y_G', 'accel_z_G', 'accel_total_G']]
            
            return synced_df[['accel_x_G', 'accel_y_G', 'accel_z_G', 'accel_total_G']].values
            
        except Exception as e:
            #print(f"Error loading sensor data {sensor_path}: {e}")
            return empty_sensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        sensor_path = self.sensor_paths[idx]
        video_id = self.metadata_df.iloc[idx]['id']
        
        # Get original class label (keeping the original labels)
        target_str = self.metadata_df.iloc[idx]['video_type']
        target = self.class_mapping.get(target_str, target_str)
        
        # Load video
        try:
            # Load video
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            num_frames = len(vr)
            video_height, video_width = vr[0].shape[:2]
            
            # Calculate required frames
            frames_needed = self.fps * self.duration
            
            # Determine sampling strategy
            if self.sample_strategy == 'metadata_time':
                # Get timestamp from metadata
                if self.time_column in self.metadata_df.columns:
                    # Get the timestamp in seconds from metadata
                    timestamp_sec = self.metadata_df.iloc[idx][self.time_column]
                    
                    # Convert timestamp to frame number
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if fps > 0:
                        frames_half = frames_needed // 2
                        center_frame = int(timestamp_sec * fps)
                        start_frame = max(0, center_frame - frames_half)
                        
                        if start_frame + frames_needed > num_frames:
                            start_frame = max(0, num_frames - frames_needed)
                        
                        start_frame = max(0, min(start_frame, num_frames - 1))
                    else:
                        start_frame = random.randint(0, max(0, num_frames - frames_needed))
                else:
                    start_frame = random.randint(0, max(0, num_frames - frames_needed))
            elif self.sample_strategy == 'center':
                # Extract frames centered around the middle of the video
                if num_frames > frames_needed:
                    # Calculate the center frame
                    center_frame = num_frames // 2
                    # Calculate half of the frames needed
                    frames_half = frames_needed // 2
                    # Start from center - half of frames needed
                    start_frame = max(0, center_frame - frames_half)
                    
                    # Make sure we don't go beyond the video length
                    if start_frame + frames_needed > num_frames:
                        start_frame = max(0, num_frames - frames_needed)
                else:
                    start_frame = 0
            else:
                # Extract from random position (default 'random' strategy)
                if num_frames > frames_needed:
                    start_frame = random.randint(0, num_frames - frames_needed)
                else:
                    start_frame = 0
                        
            # Ensure start_frame is within bounds
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = min(start_frame + frames_needed, num_frames)
            
            # Get frame indices
            indices = list(range(start_frame, end_frame))
            
            # Extract frames
            frames = vr.get_batch(indices)
            
            # Convert to numpy if needed
            if hasattr(frames, 'asnumpy'):
                frames = frames.asnumpy()
            
            # Ensure we have exactly the right number of frames
            if len(frames) < frames_needed:
                # Pad with repeated last frame
                last_frame = frames[-1] if len(frames) > 0 else np.zeros((video_height, video_width, 3), dtype=np.uint8)
                padding = np.repeat(last_frame[np.newaxis, :], frames_needed - len(frames), axis=0)
                frames = np.concatenate([frames, padding], axis=0)
            elif len(frames) > frames_needed:
                frames = frames[:frames_needed]
            
            # Convert to torch tensor 
            frames = torch.from_numpy(frames)
            
            # Reshape to [C, T, H, W] for the transform
            frames = frames.permute(3, 0, 1, 2)
            
            # Apply transform if available
            if self.transform:
                frames = self.transform(frames)
            else:
                # If no transform, just normalize to [0, 1]
                frames = frames.float() / 255.0
            
            # Convert back to [T, H, W, C] for compatibility
            frames = frames.permute(1, 2, 3, 0)
            
            # Load sensor data
            sensor_data = self._load_and_sync_sensor_data(video_path, sensor_path)
            
            # Synchronize sensor data with the selected frames
            if isinstance(sensor_data, np.ndarray) and len(sensor_data) > 0:
                if len(sensor_data) >= num_frames:
                    synced_sensor = sensor_data[start_frame:end_frame]
                    
                    # Pad or trim if needed
                    if len(synced_sensor) < frames_needed:
                        # Pad with last values
                        last_sensor = synced_sensor[-1] if len(synced_sensor) > 0 else np.zeros(4, dtype=np.float32)
                        padding = np.repeat(last_sensor[np.newaxis, :], frames_needed - len(synced_sensor), axis=0)
                        synced_sensor = np.concatenate([synced_sensor, padding], axis=0)
                    elif len(synced_sensor) > frames_needed:
                        synced_sensor = synced_sensor[:frames_needed]
                else:
                    # Not enough sensor data, use zeros
                    synced_sensor = np.zeros((frames_needed, 4), dtype=np.float32)
            else:
                # No sensor data, use zeros
                synced_sensor = np.zeros((frames_needed, 4), dtype=np.float32)
            
            # Convert sensor data to tensor
            synced_sensor = torch.from_numpy(synced_sensor).float()
            
        except Exception as e:
            #print(f"Error loading video or sensor data for {video_path}: {e}")
            # Return empty frames and sensor data with standard size
            channels = 3
            if self.transform:
                final_size = 224
                frames = torch.zeros(self.fps * self.duration, final_size, final_size, channels)
            else:
                frames = torch.zeros(self.fps * self.duration, 720, 1280, channels)
            
            synced_sensor = torch.zeros(self.fps * self.duration, 4)
        
        return {
            'frames': frames,  # Shape: [T, H, W, C]
            'sensor': synced_sensor,  # Shape: [T, 4] - (accel_x, accel_y, accel_z, accel_total)
            'target': target,
            'id': video_id
        }
    
    def show_batch(self, batch=None, m=4, figsize=(16, 12), fps=10, normalize=True, 
                  title_prefix="", temp_dir="./temp_videos", rows_per_page=2, 
                  save_videos=False, output_dir=None, video_width=240, 
                  show_sensor=True, **kwargs):
        """
        Display a batch of videos in a grid layout using actual video players, with optional sensor data visualization.
        
        Args:
            batch: Batch of data to display (if None, will sample from dataset)
            m: Number of videos per row in the grid
            figsize: Figure size for matplotlib plots
            fps: Frames per second for the output videos
            normalize: Whether to denormalize the frames for display
            title_prefix: Text to prepend to video titles
            temp_dir: Directory to store temporary video files
            rows_per_page: Number of rows in the video grid
            save_videos: Whether to save the videos permanently
            output_dir: Directory to save videos if save_videos is True
            video_width: Width of videos in the HTML display
            show_sensor: Whether to show sensor data visualization below each frame
            **kwargs: Additional arguments
            
        Returns:
            HTML display object or list of video paths if save_videos=True
        """
        import os
        import torch
        import numpy as np
        import math
        import uuid
        from pathlib import Path
        from IPython.display import HTML, display
        from torch.utils.data import DataLoader
        import imageio
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import cv2
        
        # Create directories
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        if save_videos:
            output_dir = Path(output_dir or "video_outputs")
            output_dir.mkdir(exist_ok=True)
        
        # Create a unique subfolder to avoid conflicts
        session_id = str(uuid.uuid4())[:8]
        session_dir = temp_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Get batch if not provided
        if batch is None:
            temp_loader = DataLoader(
                self, batch_size=min(m*rows_per_page, len(self)), 
                shuffle=True, num_workers=0
            )
            batch = next(iter(temp_loader))
        
        frames = batch['frames']
        sensors = batch['sensor'] if show_sensor else None
        targets = batch['target']
        ids = batch['id']
        
        # Convert tensors to numpy arrays if needed
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if show_sensor and isinstance(sensors, torch.Tensor):
            sensors = sensors.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        
        # Calculate grid dimensions
        n_videos = min(len(frames), m * rows_per_page)
        
        # Video data: paths and titles
        video_paths = []
        video_titles = []
        
        # Define mean and std used in the transforms
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        
        for i in range(n_videos):
            # Get video frames and sensor data
            video_frames = frames[i].copy()
            sensor_data = sensors[i].copy() if show_sensor else None
            
            # PROPER DENORMALIZATION
            if normalize and (video_frames.min() < 0 or video_frames.max() > 1.0):
                mean_reshaped = mean.reshape(1, 1, 1, 3)
                std_reshaped = std.reshape(1, 1, 1, 3)
                video_frames = video_frames * std_reshaped + mean_reshaped
                video_frames = np.clip(video_frames, 0, 1)
                video_frames = (video_frames * 255).astype(np.uint8)
            elif normalize and video_frames.max() <= 1.0:
                video_frames = (video_frames * 255).astype(np.uint8)
            elif video_frames.max() > 1.0:
                video_frames = video_frames.astype(np.uint8)
            
            # Ensure frames have correct shape
            if len(video_frames.shape) == 3:
                video_frames = np.repeat(video_frames[..., np.newaxis], 3, axis=-1)
            elif video_frames.shape[-1] == 1:
                video_frames = np.repeat(video_frames, 3, axis=-1)
            
            # Create a title with proper class names
            target_value = targets[i]
            if isinstance(target_value, (np.ndarray, list, tuple)) and len(target_value) > 0:
                # If target is an array, take the first element
                target_value = target_value[0]
            
            class_name = str(target_value)
                
            title = f"{title_prefix} {class_name}"
            if hasattr(ids, '__iter__'):
                if isinstance(ids[i], (str, int, float)):
                    title += f" (ID: {ids[i]})"
                else:
                    title += f" (ID: {ids[i][0] if len(ids[i]) > 0 else 'Unknown'})"
            video_titles.append(title)
            
            # Process frames - with or without sensor data
            if show_sensor:
                frames_with_sensor = []
                for j, frame in enumerate(video_frames):
                    # Create a matplotlib figure for sensor data
                    fig, ax = plt.subplots(figsize=(6, 2))
                    
                    # Plot sensor data
                    if j < len(sensor_data):
                        # Plot sensor data up to current frame
                        x_vals = np.arange(min(j+1, len(sensor_data)))
                        ax.plot(x_vals, sensor_data[:j+1, 0], 'r-', label='X')
                        ax.plot(x_vals, sensor_data[:j+1, 1], 'g-', label='Y')
                        ax.plot(x_vals, sensor_data[:j+1, 2], 'b-', label='Z')
                        ax.plot(x_vals, sensor_data[:j+1, 3], 'k-', label='Total')
                        ax.set_title('Accelerometer Data')
                        ax.legend(loc='upper right', fontsize='small')
                        ax.set_ylim(-2, 2)  # Set y-axis limits
                        ax.grid(True)
                    
                    # Convert the matplotlib figure to an image
                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    sensor_img = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
                    plt.close(fig)
                    
                    # Resize the sensor image to match the width of the video frame
                    new_height = int(frame.shape[1] * sensor_img.shape[0] / sensor_img.shape[1])
                    # Ensure that combined height (frame + sensor) is even
                    if (frame.shape[0] + new_height) % 2 != 0:
                        new_height -= 1  # Reduce sensor graph height by 1 if needed
                    
                    sensor_img = cv2.resize(sensor_img, (frame.shape[1], new_height))
                    
                    # Create a combined image with video frame on top and sensor plot below
                    combined_img = np.vstack([frame, sensor_img])
                    frames_with_sensor.append(combined_img)
                
                # Use frames with sensor data for the video
                processed_frames = frames_with_sensor
            else:
                # Make sure frames have even dimensions for FFMPEG
                processed_frames = []
                for frame in video_frames:
                    height, width = frame.shape[:2]
                    # Ensure even dimensions by padding if necessary
                    if height % 2 != 0 or width % 2 != 0:
                        new_height = height + (height % 2)
                        new_width = width + (width % 2)
                        # Create a new frame with black padding
                        padded_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                        padded_frame[:height, :width] = frame
                        processed_frames.append(padded_frame)
                    else:
                        processed_frames.append(frame)
            
            # Define video filename
            temp_video_path = str(session_dir / f"video_{i}.mp4")
            video_paths.append(temp_video_path)
            
            # Save video file using imageio
            try:
                imageio.mimwrite(temp_video_path, processed_frames, fps=fps, macro_block_size=1)
                
                if save_videos:
                    output_video_path = str(output_dir / f"{title.replace(' ', '_')}_{i}.mp4")
                    import shutil
                    shutil.copy(temp_video_path, output_video_path)
                    
            except Exception as e:
                print(f"Error creating video {i}: {e}")
                print(f"Frame dimensions: {processed_frames[0].shape}")
                if isinstance(e, BrokenPipeError):
                    # Try to get FFMPEG error details
                    import subprocess
                    try:
                        # Test FFMPEG with a simple command
                        result = subprocess.run(
                            ["ffmpeg", "-version"], 
                            capture_output=True, 
                            text=True
                        )
                        print(f"FFMPEG version info: {result.stdout[:100]}...")
                    except Exception as ffmpeg_err:
                        print(f"FFMPEG test error: {ffmpeg_err}")
        
        # Create HTML for grid layout
        html = """
        <style>
        .video-grid {
            display: grid;
            grid-template-columns: repeat(AUTO_COLUMNS, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        .video-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 5px;
        }
        .video-title {
            text-align: center;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 14px;
            height: 40px;
            overflow: hidden;
        }
        video {
            width: VIDEOWIDTHpx;
            border: 1px solid #ddd;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .normal { color: #2ca02c; }
        .collision { color: #d62728; }
        .near-collision { color: #ff7f0e; }
        </style>
        """.replace("AUTO_COLUMNS", str(m)).replace("VIDEOWIDTH", str(video_width))
        
        # Add batch heading
        html += f"<h2>Video Batch{' with Sensor Data' if show_sensor else ''}</h2>"
        html += "<div class='video-grid'>"
        
        # Check if we're in Jupyter/IPython environment
        is_jupyter = True
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                is_jupyter = False
        except ImportError:
            is_jupyter = False
        
        # Add videos to grid
        for i, (path, title) in enumerate(zip(video_paths, video_titles)):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                html += "<div class='video-container'>"
                
                # Add title with color coding for class
                if "Normal" in title:
                    class_type = "normal"
                elif "Near Collision" in title:
                    class_type = "near-collision"
                else:
                    class_type = "collision"
                    
                html += f"<div class='video-title {class_type}'>{title}</div>"
                
                # Base64 encode the video for more reliable display
                try:
                    from base64 import b64encode
                    mp4 = open(path, 'rb').read()
                    data_url = f"data:video/mp4;base64,{b64encode(mp4).decode()}"
                    
                    html += f"""
                    <video controls autoplay loop muted playsinline>
                        <source src="{data_url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """
                except Exception as e:
                    # Fallback to file URL (less reliable)
                    html += f"""
                    <video controls loop muted playsinline>
                        <source src="file://{path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """
                
                html += "</div>"
        
        html += "</div>"
        
        # Display HTML in Jupyter
        display_html = HTML(html)
        if is_jupyter:
            display(display_html)
        else:
            print(f"Created {len(video_paths)} videos in {session_dir}")
            
        # Clean up temporary files if not saving
        if not save_videos:
            import shutil
            try:
                # Wait a bit to ensure videos can be loaded
                import time
                time.sleep(1)
                shutil.rmtree(session_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {e}")
        
        # Return paths if saving videos, otherwise return HTML
        if save_videos:
            return video_paths
        else:
            return display_html


def create_datasets_with_multiple_dirs(base_dirs, metadata_csv, seed=42, sensor_subdir='signals',
                                   sample_strategy='random', time_column=None, show_stats=False):
    """Create datasets with support for multiple directories and IMU data
    
    Args:
        base_dirs: List of base directories to search for videos
        metadata_csv: Path to CSV file with video metadata
        seed: Random seed for reproducibility
        sensor_subdir: Subdirectory name containing sensor data
        sample_strategy: Strategy for sampling frames - 'random', 'metadata_time', or 'center'
        time_column: Column name in metadata containing timestamp for 'metadata_time' strategy
        show_stats: Whether to display visual statistics about the dataset at the end
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Make sure base_dirs is a list
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    # Validate sample_strategy
    if sample_strategy not in ['random', 'metadata_time', 'center']:
        print(f"Warning: Invalid sample strategy '{sample_strategy}'. Using 'random' instead.")
        sample_strategy = 'random'
    
    if sample_strategy == 'metadata_time' and time_column is None:
        print(f"Warning: 'metadata_time' strategy requires time_column. Using 'random' instead.")
        sample_strategy = 'random'
    
    # 1. Load ALL metadata
    df = pd.read_csv(metadata_csv)
    original_count = len(df)
    original_distribution = df['video_type'].value_counts()
    
    # 2. Check which files exist across all directories
    existing_mask = []
    found_by_dir = {i: 0 for i in range(len(base_dirs))}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        video_id = row['id']
        found = False
        
        # Check each directory
        for dir_idx, base_dir in enumerate(base_dirs):
            video_dir = os.path.join(base_dir, video_id)
            if not os.path.exists(video_dir):
                continue
                
            files = os.listdir(video_dir)
            video_file = None
            
            for file in files:
                if file.endswith('.mp4') or file.endswith('.mov'):
                    video_file = os.path.join(video_dir, file)
                    break
                    
            sensor_file = os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
            
            if video_file and os.path.exists(video_file) and os.path.exists(sensor_file):
                found = True
                found_by_dir[dir_idx] += 1
                break
        
        existing_mask.append(found)
    
    # 3. Filter to only existing files
    existing_mask = np.array(existing_mask)
    existing_df = df[existing_mask].copy()
    existing_distribution = existing_df['video_type'].value_counts()
    
    # 4. Remove classes with too few samples for stratified split
    min_samples_per_class = 5
    valid_classes = existing_df['video_type'].value_counts()
    valid_classes = valid_classes[valid_classes >= min_samples_per_class].index
    
    if len(valid_classes) < len(existing_df['video_type'].value_counts()):
        print(f"\n⚠️  Warning: Removing classes with less than {min_samples_per_class} samples:")
        removed_classes = existing_df['video_type'].value_counts()
        removed_classes = removed_classes[removed_classes < min_samples_per_class]
        for class_name, count in removed_classes.items():
            print(f"  - {class_name}: {count} samples")
    
    # Filter to only valid classes
    filtered_df = existing_df[existing_df['video_type'].isin(valid_classes)].copy()
    filtered_distribution = filtered_df['video_type'].value_counts()
    
    # 5. NOW do the stratified split
    train_df, temp_df = train_test_split(
        filtered_df, 
        test_size=0.3, 
        random_state=seed, 
        stratify=filtered_df['video_type']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=seed, 
        stratify=temp_df['video_type']
    )
    
    train_distribution = train_df['video_type'].value_counts()
    val_distribution = val_df['video_type'].value_counts()
    test_distribution = test_df['video_type'].value_counts()
    
    # Create datasets
    train_data = NvidiaDashcamDataset(
        train_df, 
        base_dirs, 
        is_train=True, 
        skip_missing=True, 
        transform=create_video_transforms(mode='train'),
        sensor_subdir=sensor_subdir,
        sample_strategy=sample_strategy,
        time_column=time_column
    )
    
    val_data = NvidiaDashcamDataset(
        val_df, 
        base_dirs, 
        is_train=False, 
        skip_missing=True, 
        transform=create_video_transforms(mode='val'),
        sensor_subdir=sensor_subdir,
        sample_strategy=sample_strategy,
        time_column=time_column
    )
    
    test_data = NvidiaDashcamDataset(
        test_df, 
        base_dirs, 
        is_train=False, 
        skip_missing=True, 
        transform=create_video_transforms(mode='val'),
        sensor_subdir=sensor_subdir,
        sample_strategy=sample_strategy,
        time_column=time_column
    )
    
    # Display visual statistics if requested
    if show_stats:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display, HTML
        
        # Set the style
        plt.style.use('fivethirtyeight')
        sns.set_palette('colorblind')
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("NVIDIA Dashcam Dataset Statistics", fontsize=20, fontweight='bold')
        
        # 1. Distribution comparison - Original vs. Filtered
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        
        all_classes = sorted(set(original_distribution.index) | set(filtered_distribution.index))
        x = np.arange(len(all_classes))
        width = 0.35
        
        # Get data for each distribution, filling in zeros for missing classes
        orig_data = [original_distribution.get(cls, 0) for cls in all_classes]
        filt_data = [filtered_distribution.get(cls, 0) for cls in all_classes]
        
        # Plot bars
        ax1.bar(x - width/2, orig_data, width, label='Original')
        ax1.bar(x + width/2, filt_data, width, label='Filtered')
        
        # Add labels and legend
        ax1.set_title('Class Distribution: Original vs. Filtered')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_classes, rotation=45, ha='right')
        ax1.legend()
        
        # Add counts on bars
        for i, v in enumerate(orig_data):
            ax1.text(i - width/2, v + 100, f"{v}", ha='center', fontsize=9)
        for i, v in enumerate(filt_data):
            ax1.text(i + width/2, v + 100, f"{v}", ha='center', fontsize=9)
            
        # 2. Files found by directory
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        dir_names = [os.path.basename(d) for d in base_dirs]
        counts = list(found_by_dir.values())
        
        ax2.pie(counts, labels=dir_names, autopct='%1.1f%%', startangle=90, shadow=True)
        ax2.set_title('Files Found by Directory')
        
        # 3. Train/Val/Test Split Distribution
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        
        # Get data
        splits = ["Train", "Validation", "Test"]
        distributions = [train_distribution, val_distribution, test_distribution]
        
        # Create a bar for each class in each split
        for i, cls in enumerate(filtered_distribution.index):
            class_data = [dist.get(cls, 0) for dist in distributions]
            offset = (i - len(filtered_distribution.index)/2 + 0.5) * width
            ax3.bar(np.arange(len(splits)) + offset, class_data, width, label=cls)
        
        # Add labels and legend
        ax3.set_title('Train/Val/Test Split by Class')
        ax3.set_xticks(np.arange(len(splits)))
        ax3.set_xticklabels(splits)
        ax3.legend(loc='upper right')
        
        # Add percentage annotations
        for i, split_df in enumerate([train_df, val_df, test_df]):
            total = len(split_df)
            y_offset = 0
            for cls in filtered_distribution.index:
                count = split_df['video_type'].value_counts().get(cls, 0)
                percentage = count / total * 100
                
                # Find the class position
                cls_idx = list(filtered_distribution.index).index(cls)
                bar_x = i + (cls_idx - len(filtered_distribution.index)/2 + 0.5) * width
                
                ax3.text(bar_x, count + 100, f"{percentage:.1f}%", ha='center', fontsize=8, rotation=90)
        
        # Create a text box with summary statistics
        stats_text = f"""
        <div style="font-family: Arial; background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin: 20px;">
            <h2 style="color: #333; text-align: center;">Dataset Summary</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #ddd;">
                    <th style="text-align: left; padding: 8px;">Metric</th>
                    <th style="text-align: right; padding: 8px;">Value</th>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Original records</td>
                    <td style="text-align: right; padding: 8px;">{original_count}</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Valid records found</td>
                    <td style="text-align: right; padding: 8px;">{len(filtered_df)} ({len(filtered_df)/original_count*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Records filtered out</td>
                    <td style="text-align: right; padding: 8px;">{original_count - len(filtered_df)} ({(original_count - len(filtered_df))/original_count*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Training set size</td>
                    <td style="text-align: right; padding: 8px;">{len(train_df)} ({len(train_df)/len(filtered_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Validation set size</td>
                    <td style="text-align: right; padding: 8px;">{len(val_df)} ({len(val_df)/len(filtered_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Test set size</td>
                    <td style="text-align: right; padding: 8px;">{len(test_df)} ({len(test_df)/len(filtered_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Sampling strategy</td>
                    <td style="text-align: right; padding: 8px;">{sample_strategy}</td>
                </tr>
            </table>
        </div>
        """
        
        display(HTML(stats_text))
        
        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # Display class distribution details
        class_stats_html = f"""
        <div style="font-family: Arial; background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin: 20px;">
            <h2 style="color: #333; text-align: center;">Class Distribution Details</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #ddd; background-color: #f2f2f2;">
                    <th style="text-align: left; padding: 8px;">Class</th>
                    <th style="text-align: right; padding: 8px;">Original</th>
                    <th style="text-align: right; padding: 8px;">Filtered</th>
                    <th style="text-align: right; padding: 8px;">Training</th>
                    <th style="text-align: right; padding: 8px;">Validation</th>
                    <th style="text-align: right; padding: 8px;">Test</th>
                </tr>
        """
        
        for cls in sorted(all_classes):
            orig = original_distribution.get(cls, 0)
            filt = filtered_distribution.get(cls, 0)
            train = train_distribution.get(cls, 0)
            val = val_distribution.get(cls, 0)
            test = test_distribution.get(cls, 0)
            
            # Calculate retention percentage
            retention = (filt / orig * 100) if orig > 0 else 0
            
            # Color code based on retention
            color = "#d4edda"  # Green for good retention
            if retention < 90:
                color = "#fff3cd"  # Yellow for moderate retention
            if retention < 70:
                color = "#f8d7da"  # Red for poor retention
                
            class_stats_html += f"""
                <tr style="border-bottom: 1px solid #ddd; background-color: {color};">
                    <td style="padding: 8px;"><b>{cls}</b></td>
                    <td style="text-align: right; padding: 8px;">{orig}</td>
                    <td style="text-align: right; padding: 8px;">{filt} ({retention:.1f}%)</td>
                    <td style="text-align: right; padding: 8px;">{train} ({train/filt*100:.1f}%)</td>
                    <td style="text-align: right; padding: 8px;">{val} ({val/filt*100:.1f}%)</td>
                    <td style="text-align: right; padding: 8px;">{test} ({test/filt*100:.1f}%)</td>
                </tr>
            """
        
        class_stats_html += """
            </table>
        </div>
        """
        
        display(HTML(class_stats_html))
    
    return train_data, val_data, test_data

import os
from pathlib import Path
from typing import List, Tuple, Optional

def find_video_path(video_id: str, base_dirs: List[str], 
                   check_sensors: bool = True, 
                   sensor_subdir: str = 'signals') -> dict:
    """
    Find the full path to video and sensor files for a given video ID across multiple directories.
    
    Args:
        video_id: The ID of the video to find
        base_dirs: List of base directories to search in
        check_sensors: Whether to check for sensor files as well
        sensor_subdir: Subdirectory name containing sensor files
        
    Returns:
        Dictionary containing paths and status information:
        {
            'found': bool,
            'video_path': str or None,
            'sensor_path': str or None,
            'directory': str or None,
            'video_format': str or None,
            'message': str
        }
    """
    # Possible video filename formats based on different naming conventions
    video_formats = [
        f"{video_id}.mp4",                   # Format in nvidia-2
        f"anonymized_{video_id}.mp4",        # Format in nvidia-1
        f"{video_id}.mov",                   # Alternative format
        f"dash_{video_id}.mp4",              # Another possible format
        f"video_{video_id}.mp4",             # Another possible format
        f"dashcam_{video_id}.mp4"            # Another possible format
    ]
    
    # Sensor filename
    sensor_file = "Dashcam-Accelerometer_Acceleration.csv"
    
    # Initialize result
    result = {
        'found': False,
        'video_path': None,
        'sensor_path': None,
        'directory': None,
        'video_format': None,
        'message': f"Video ID '{video_id}' not found in any directory."
    }
    
    # Search in all provided directories
    for base_dir in base_dirs:
        # Full path to the video's directory
        video_dir = os.path.join(base_dir, video_id)
        
        # Check if the directory exists
        if not os.path.isdir(video_dir):
            continue
        
        # Try all possible video formats
        for format_name in video_formats:
            video_path = os.path.join(video_dir, format_name)
            
            if os.path.exists(video_path):
                result['found'] = True
                result['video_path'] = video_path
                result['directory'] = base_dir
                result['video_format'] = format_name
                
                # Check for sensor data if requested
                if check_sensors:
                    sensor_path = os.path.join(video_dir, sensor_subdir, sensor_file)
                    if os.path.exists(sensor_path):
                        result['sensor_path'] = sensor_path
                        result['message'] = f"Found video and sensor data in {base_dir}"
                    else:
                        result['message'] = f"Found video in {base_dir} but sensor data is missing"
                else:
                    result['message'] = f"Found video in {base_dir}"
                
                return result
    
    # If video directory exists but no matching video file
    for base_dir in base_dirs:
        video_dir = os.path.join(base_dir, video_id)
        if os.path.isdir(video_dir):
            # Directory exists but video file not found with any known format
            result['message'] = f"Directory exists in {base_dir} but no matching video file found. Contents: {os.listdir(video_dir)}"
            return result
    
    # Return the default "not found" result
    return result

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import cv2

def add_peak_acceleration_timestamps(metadata_df, base_dirs, sensor_subdir='signals', output_path=None,
                                     try_alternative_formats=True):
    """
    Process all videos in the metadata and add a column with the timestamp of peak acceleration.
    Handles both nvidia-1 and nvidia-2 CSV formats correctly.
    
    Args:
        metadata_df: DataFrame with video metadata
        base_dirs: List of base directories where videos are stored
        sensor_subdir: Subdirectory for sensor data
        output_path: Path to save the updated metadata (optional)
        try_alternative_formats: Whether to try alternative video file formats
        
    Returns:
        DataFrame with added 'peak_accel_time_sec' column
    """
    # Make sure base_dirs is a list
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    # Create a copy of the metadata to avoid modifying the original
    metadata_with_peaks = metadata_df.copy()
    
    # Add a new column for peak acceleration timestamp
    metadata_with_peaks['peak_accel_time_sec'] = np.nan
    
    # Counters for statistics
    processed_count = 0
    missing_count = 0
    error_count = 0
    
    # Format statistics
    format_counts = {
        "standard": 0,
        "anonymized": 0,
        "alternative": 0
    }
    
    # Schema tracking
    schema_counts = {
        "nvidia-1": 0,
        "nvidia-2": 0
    }
    
    ##print(f"Processing {len(metadata_df)} videos to find peak acceleration timestamps...")
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Finding peak acceleration"):
        video_id = row['id']
        
        # Initialize found flags
        video_found = False
        sensor_found = False
        sensor_path = None
        video_file = None
        format_found = None
        
        # Try to find the video and sensor data in any of the base directories
        for base_dir in base_dirs:
            # Define all possible primary formats to check - we'll check all without assumptions
            primary_formats = [
                {
                    "name": "standard",
                    "video": os.path.join(base_dir, video_id, f"{video_id}.mp4"),
                    "sensor": os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
                },
                {
                    "name": "anonymized",
                    "video": os.path.join(base_dir, video_id, f"anonymized_{video_id}.mp4"),
                    "sensor": os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
                }
            ]
            
            # Check all primary formats first
            for format_spec in primary_formats:
                if os.path.exists(format_spec["video"]) and os.path.exists(format_spec["sensor"]):
                    video_found = True
                    sensor_found = True
                    video_file = format_spec["video"]
                    sensor_path = format_spec["sensor"]
                    format_found = format_spec["name"]
                    break
            
            # If we found both, exit the loop
            if video_found and sensor_found:
                break
                
            # Try alternative formats if enabled and primary formats not found
            if try_alternative_formats and (not video_found or not sensor_found):
                # Alternative video formats
                alternative_formats = [
                    os.path.join(base_dir, f"{video_id}.mp4"),  # Flat structure
                    os.path.join(base_dir, "videos", f"{video_id}.mp4"),  # Videos subfolder
                    os.path.join(base_dir, video_id, "video", f"{video_id}.mp4"),  # Video subfolder
                    os.path.join(base_dir, video_id, f"video_{video_id}.mp4")  # Other naming convention
                ]
                
                # Alternative sensor paths
                alternative_sensor_paths = [
                    os.path.join(base_dir, video_id, "sensor", "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, video_id, "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, "sensors", video_id, "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, sensor_subdir, video_id, "Dashcam-Accelerometer_Acceleration.csv")
                ]
                
                # Check alternative video formats
                for alt_video in alternative_formats:
                    if os.path.exists(alt_video):
                        # Try each alternative sensor path
                        for alt_sensor in alternative_sensor_paths:
                            if os.path.exists(alt_sensor):
                                video_found = True
                                sensor_found = True
                                video_file = alt_video
                                sensor_path = alt_sensor
                                format_found = "alternative"
                                break
                        
                        # If we found video but no sensor, keep looking
                        if video_found and not sensor_found:
                            continue
                        # If we found both, exit the loop
                        elif video_found and sensor_found:
                            break
        
        # Update format statistics
        if format_found:
            format_counts[format_found] += 1
        
        # Debug output for troubleshooting
        # if idx < 5 or idx % 100 == 0:  # Show for first 5 items and then every 100th item
        #     #print(f"Video {idx} (id={video_id}):")
        #     #print(f"  Video found: {video_found} - {video_file if video_found else 'Not found'}")
        #     #print(f"  Sensor found: {sensor_found} - {sensor_path if sensor_found else 'Not found'}")
        #     if format_found:
        #         #print(f"  Format: {format_found}")
        
        if not video_found or not sensor_found:
            missing_count += 1
            continue
        
        try:
            # Read the first line to check the format
            with open(sensor_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Determine format based on header
            is_nvidia1 = "Dashcam-Accelerometer.Acceleration" in first_line
            
            if is_nvidia1:
                # This is nvidia-1 format
                schema_counts["nvidia-1"] += 1
                
                # For nvidia-1, we need to parse the csv correctly
                # The columns are comma-separated, and the header contains the column names
                column_names = first_line.split(',')
                
                # Read the data with the correct column names
                df_sensor = pd.read_csv(sensor_path, names=column_names, skiprows=1)
                
                # Define the column mappings
                time_col = column_names[0]  # First column is timestamp
                accel_x_col = column_names[1]  # Second column is x acceleration
                accel_y_col = column_names[2]  # Third column is y acceleration
                accel_z_col = column_names[3]  # Fourth column is z acceleration
                
            else:
                # This is nvidia-2 format (standard CSV)
                schema_counts["nvidia-2"] += 1
                
                # Read the CSV normally
                df_sensor = pd.read_csv(sensor_path)
                
                # Define standard column names
                time_col = 'time_sec'
                accel_x_col = 'accel_x_G'
                accel_y_col = 'accel_y_G'
                accel_z_col = 'accel_z_G'
                accel_total_col = 'accel_total_G' if 'accel_total_G' in df_sensor.columns else None
            
            # # #print debugging info for the first few files
            # if idx < 5:
            #     #print(f"  CSV Format detected: {'nvidia-1' if is_nvidia1 else 'nvidia-2'}")
            #     #print(f"  Column names: {df_sensor.columns.tolist()}")
            #     #print(f"  Time column: {time_col}")
            #     if len(df_sensor) > 0:
            #         #print(f"  Sample data:\n{df_sensor.head(2)}")
            
            # Calculate total acceleration if needed
            accel_total_col = 'accel_total_G'
            if accel_total_col not in df_sensor.columns:
                # Calculate the total acceleration
                df_sensor[accel_total_col] = np.sqrt(
                    df_sensor[accel_x_col]**2 + 
                    df_sensor[accel_y_col]**2 + 
                    df_sensor[accel_z_col]**2
                )
            
            # Find peak acceleration
            max_idx = df_sensor[accel_total_col].idxmax()
            max_time = df_sensor.loc[max_idx, time_col]
            
            # Store in the metadata
            metadata_with_peaks.at[idx, 'peak_accel_time_sec'] = max_time
            processed_count += 1
                
        except Exception as e:
            #print(f"Error processing sensor data for {video_id}: {e}")
            # Try to #print the actual file structure for debugging
            try:
                with open(sensor_path, 'r') as f:
                    first_lines = [next(f) for _ in range(5)]
                #print(f"  First 5 lines of file {video_id}:")
                for i, line in enumerate(first_lines):
                    print(f"    {i}: {line.strip()}")
            except:
                pass
            error_count += 1
    
    # #print summary statistics
    # #print(f"\nProcessing complete:")
    # #print(f"  Successfully processed: {processed_count} videos")
    # #print(f"  Missing videos/sensors: {missing_count} videos")
    # #print(f"  Errors encountered: {error_count} videos")
    # #print(f"  Formats found: standard={format_counts['standard']}, " 
    #       f"anonymized={format_counts['anonymized']}, alternative={format_counts['alternative']}")
    # #print(f"  Schema formats: nvidia-1={schema_counts['nvidia-1']}, nvidia-2={schema_counts['nvidia-2']}")
    
    # Save the updated metadata if output path is provided
    if output_path:
        metadata_with_peaks.to_csv(output_path, index=False)
        ##print(f"Updated metadata saved to: {output_path}")
    
    return metadata_with_peaks

def convert_absolute_to_relative_time(metadata_df, base_dirs, sensor_subdir='signals', output_path=None,
                                     try_alternative_formats=True):
    """
    Convert absolute peak acceleration timestamps to relative timestamps from video start.
    Handles both nvidia-1 and nvidia-2 CSV formats correctly.
    
    Args:
        metadata_df: DataFrame with video metadata containing 'peak_accel_time_sec'
        base_dirs: List of base directories where videos are stored
        sensor_subdir: Subdirectory for sensor data
        output_path: Path to save the updated metadata (optional)
        try_alternative_formats: Whether to try alternative video file formats
        
    Returns:
        DataFrame with added 'peak_accel_rel_time_sec' column
    """
    # Make sure base_dirs is a list
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    # Create a copy of the metadata to avoid modifying the original
    metadata_with_rel_peaks = metadata_df.copy()
    
    # Add a new column for relative peak acceleration timestamp
    metadata_with_rel_peaks['peak_accel_rel_time_sec'] = np.nan
    
    # Counters for statistics
    processed_count = 0
    missing_count = 0
    error_count = 0
    no_peak_count = 0
    
    # Format statistics
    format_counts = {
        "standard": 0,
        "anonymized": 0,
        "alternative": 0
    }
    
    # Schema tracking
    schema_counts = {
        "nvidia-1": 0,
        "nvidia-2": 0
    }
    
    ##print(f"Processing {len(metadata_df)} videos to convert absolute to relative timestamps...")
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Converting timestamps"):
        video_id = row['id']
        
        # Skip if peak acceleration time is not available
        if pd.isna(row.get('peak_accel_time_sec')):
            no_peak_count += 1
            continue
            
        peak_abs_time = row['peak_accel_time_sec']
        
        # Initialize found flags
        video_found = False
        sensor_found = False
        sensor_path = None
        video_file = None
        format_found = None
        
        # Try to find the video and sensor data in any of the base directories
        for base_dir in base_dirs:
            # Define all possible primary formats to check - we'll check all without assumptions
            primary_formats = [
                {
                    "name": "standard",
                    "video": os.path.join(base_dir, video_id, f"{video_id}.mp4"),
                    "sensor": os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
                },
                {
                    "name": "anonymized",
                    "video": os.path.join(base_dir, video_id, f"anonymized_{video_id}.mp4"),
                    "sensor": os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
                }
            ]
            
            # Check all primary formats first
            for format_spec in primary_formats:
                if os.path.exists(format_spec["video"]) and os.path.exists(format_spec["sensor"]):
                    video_found = True
                    sensor_found = True
                    video_file = format_spec["video"]
                    sensor_path = format_spec["sensor"]
                    format_found = format_spec["name"]
                    break
            
            # If we found both, exit the loop
            if video_found and sensor_found:
                break
                
            # Try alternative formats if enabled and not found yet
            if try_alternative_formats and (not video_found or not sensor_found):
                # Alternative video formats
                alternative_formats = [
                    os.path.join(base_dir, f"{video_id}.mp4"),  # Flat structure
                    os.path.join(base_dir, "videos", f"{video_id}.mp4"),  # Videos subfolder
                    os.path.join(base_dir, video_id, "video", f"{video_id}.mp4"),  # Video subfolder
                    os.path.join(base_dir, video_id, f"video_{video_id}.mp4")  # Other naming convention
                ]
                
                # Alternative sensor paths
                alternative_sensor_paths = [
                    os.path.join(base_dir, video_id, "sensor", "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, video_id, "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, "sensors", video_id, "Dashcam-Accelerometer_Acceleration.csv"),
                    os.path.join(base_dir, sensor_subdir, video_id, "Dashcam-Accelerometer_Acceleration.csv")
                ]
                
                # Check alternative video formats
                for alt_video in alternative_formats:
                    if os.path.exists(alt_video):
                        # Try each alternative sensor path
                        for alt_sensor in alternative_sensor_paths:
                            if os.path.exists(alt_sensor):
                                video_found = True
                                sensor_found = True
                                video_file = alt_video
                                sensor_path = alt_sensor
                                format_found = "alternative"
                                break
                        
                        # If we found video but no sensor, keep looking
                        if video_found and not sensor_found:
                            continue
                        # If we found both, exit the loop
                        elif video_found and sensor_found:
                            break
        
        # Update format statistics
        if format_found:
            format_counts[format_found] += 1
            
        if not video_found or not sensor_found:
            missing_count += 1
            continue
        
        try:
            # Read the first line to check the format
            with open(sensor_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Determine format based on header
            is_nvidia1 = "Dashcam-Accelerometer.Acceleration" in first_line
            
            if is_nvidia1:
                # This is nvidia-1 format
                schema_counts["nvidia-1"] += 1
                
                # For nvidia-1, we need to parse the csv correctly
                # The columns are comma-separated, and the header contains the column names
                column_names = first_line.split(',')
                
                # Read the data with the correct column names
                df_sensor = pd.read_csv(sensor_path, names=column_names, skiprows=1)
                
                # Define the column mappings
                time_col = column_names[0]  # First column is timestamp
                
            else:
                # This is nvidia-2 format (standard CSV)
                schema_counts["nvidia-2"] += 1
                
                # Read the CSV normally
                df_sensor = pd.read_csv(sensor_path)
                
                # Define standard column names
                time_col = 'time_sec'
            
            # #print debugging info for the first few files
            # if idx < 5:
            #     #print(f"  CSV Format detected: {'nvidia-1' if is_nvidia1 else 'nvidia-2'}")
            #     #print(f"  Time column: {time_col}")
            #     if len(df_sensor) > 0:
            #         #print(f"  First time value: {df_sensor[time_col].iloc[0]}")
            
            # Get the first timestamp in the sensor data (video start)
            sensor_start_time = df_sensor[time_col].iloc[0]
            
            # Calculate relative time
            peak_rel_time = peak_abs_time - sensor_start_time
            
            # Ensure relative time is not negative
            if peak_rel_time < 0:
                #print(f"Warning: Negative relative time for {video_id}, setting to 0")
                peak_rel_time = 0
                
            # Get video length to check upper bound
            try:
                cap = cv2.VideoCapture(video_file)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                video_duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                if video_duration > 0 and peak_rel_time > video_duration:
                    #print(f"Warning: Relative time ({peak_rel_time:.2f}s) beyond video duration ({video_duration:.2f}s) for {video_id}, capping at video length")
                    peak_rel_time = video_duration
            except Exception as e:
                print(f"Warning: Could not check video duration for {video_id}: {e}")
            
            # Store in the metadata
            metadata_with_rel_peaks.at[idx, 'peak_accel_rel_time_sec'] = peak_rel_time
            processed_count += 1
                
        except Exception as e:
            print(f"Error processing sensor data for {video_id}: {e}")
            error_count += 1
    
    # # #print summary statistics
    # #print(f"\nProcessing complete:")
    # #print(f"  Successfully processed: {processed_count} videos")
    # #print(f"  Missing videos/sensors: {missing_count} videos")
    # #print(f"  No peak acceleration data: {no_peak_count} videos")
    # #print(f"  Errors encountered: {error_count} videos")
    # #print(f"  Formats found: standard={format_counts['standard']}, " 
    #       f"anonymized={format_counts['anonymized']}, alternative={format_counts['alternative']}")
    # #print(f"  Schema formats: nvidia-1={schema_counts['nvidia-1']}, nvidia-2={schema_counts['nvidia-2']}")
    
    # Save the updated metadata if output path is provided
    if output_path:
        metadata_with_rel_peaks.to_csv(output_path, index=False)
        ##print(f"Updated metadata saved to: {output_path}")
    
    return metadata_with_rel_peaks

def infer_directory_structure(base_dirs, sample_ids=None, max_samples=5):
    """
    Attempt to infer the actual directory structure for videos and sensor data.
    
    Args:
        base_dirs: List of base directories to check
        sample_ids: Optional list of video IDs to check
        max_samples: Maximum number of samples to check
        
    Returns:
        Dictionary with inferred directory structure information
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    # Check if sample_ids is None or empty (handling both lists and numpy arrays)
    if sample_ids is None or (hasattr(sample_ids, '__len__') and len(sample_ids) == 0):
        # If no sample IDs provided, try to list directories
        sample_ids = []
        for base_dir in base_dirs:
            try:
                contents = os.listdir(base_dir)
                # Take a few potential video IDs (directories or files)
                potential_ids = [item for item in contents if not item.startswith('.')][:max_samples]
                sample_ids.extend(potential_ids)
            except Exception as e:
                print(f"Could not list directory {base_dir}: {e}")
    
    # Convert numpy array to list if needed
    if hasattr(sample_ids, 'tolist'):
        sample_ids = sample_ids.tolist()
        
    # Limit the number of samples
    sample_ids = sample_ids[:max_samples]
    
    ##print(f"Checking {len(sample_ids)} sample IDs across {len(base_dirs)} directories...")
    
    # Patterns to check
    patterns = {
        "standard": {
            "video": "{base_dir}/{id}/{id}.mp4",
            "sensor": "{base_dir}/{id}/signals/Dashcam-Accelerometer_Acceleration.csv"
        },
        "anonymized": {
            "video": "{base_dir}/{id}/anonymized_{id}.mp4",
            "sensor": "{base_dir}/{id}/signals/Dashcam-Accelerometer_Acceleration.csv"
        },
        "flat": {
            "video": "{base_dir}/{id}.mp4",
            "sensor": "{base_dir}/signals/{id}/Dashcam-Accelerometer_Acceleration.csv"
        },
        "subfolder": {
            "video": "{base_dir}/videos/{id}.mp4",
            "sensor": "{base_dir}/signals/{id}/Dashcam-Accelerometer_Acceleration.csv"
        }
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        results[pattern_name] = {"count": 0, "examples": []}
    
    # Check each sample ID
    for video_id in sample_ids:
        for base_dir in base_dirs:
            for pattern_name, pattern in patterns.items():
                video_path = pattern["video"].format(base_dir=base_dir, id=video_id)
                sensor_path = pattern["sensor"].format(base_dir=base_dir, id=video_id)
                
                video_exists = os.path.exists(video_path)
                sensor_exists = os.path.exists(sensor_path)
                
                if video_exists or sensor_exists:
                    results[pattern_name]["count"] += 1
                    results[pattern_name]["examples"].append({
                        "id": video_id,
                        "base_dir": base_dir,
                        "video_path": video_path,
                        "video_exists": video_exists,
                        "sensor_path": sensor_path,
                        "sensor_exists": sensor_exists
                    })
    
    # Find the best matching pattern
    best_pattern = max(results.items(), key=lambda x: x[1]["count"])
    
    # #print("\nDirectory structure analysis:")
    # for pattern_name, data in results.items():
    #     #print(f"  {pattern_name}: {data['count']} matches")
    #     if data["count"] > 0:
    #         example = data["examples"][0]
    #         #print(f"    Example: {example['id']}")
    #         #print(f"      Video: {'✓' if example['video_exists'] else '✗'} {example['video_path']}")
    #         #print(f"      Sensor: {'✓' if example['sensor_exists'] else '✗'} {example['sensor_path']}")
    
    # #print(f"\nRecommended pattern: {best_pattern[0]}")
    
    return {
        "results": results,
        "best_pattern": best_pattern[0],
        "patterns": patterns
    }

def copy_video_file(src_path: str, dst_path: str) -> None:
    """
    Copies a video file from source to destination preserving metadata.
    
    Args:
        src_path (str): Full path to the source video file.
        dst_path (str): Full path to the destination file location.
    """
    shutil.copy2(src_path, dst_path)

import pandas as pd
import matplotlib.pyplot as plt

def plot_acceleration(csv_path: str) -> None:
    """
    Loads accelerometer data from a CSV file and plots all acceleration components over time.
    
    Args:
        csv_path (str): Path to the CSV file containing acceleration data.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df['time_sec'] = pd.to_datetime(df['time_sec'], unit='s')

    plt.figure(figsize=(14, 6))
    plt.plot(df['time_sec'], df['accel_x_G'], label='Accel X')
    plt.plot(df['time_sec'], df['accel_y_G'], label='Accel Y')
    plt.plot(df['time_sec'], df['accel_z_G'], label='Accel Z')
    plt.plot(df['time_sec'], df['accel_total_G'], label='Accel Total', linewidth=2, linestyle='--')

    plt.xlabel("Time")
    plt.ylabel("Acceleration (G)")
    plt.title("Acceleration Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_datasets_with_manual_split(base_dirs, metadata_csv, seed=42, sensor_subdir='signals',
                                    sample_strategy='random', time_column=None, show_stats=False,
                                    split_column='split', validate_split=True):
    """Create datasets using manual split specification from metadata
    
    Args:
        base_dirs: List of base directories to search for videos
        metadata_csv: Path to CSV file with video metadata
        seed: Random seed for reproducibility
        sensor_subdir: Subdirectory name containing sensor data
        sample_strategy: Strategy for sampling frames - 'random', 'metadata_time', or 'center'
        time_column: Column name in metadata containing timestamp for 'metadata_time' strategy
        show_stats: Whether to display visual statistics about the dataset at the end
        split_column: Column name in metadata specifying train/val/test split
        validate_split: Whether to validate the split has all required sets
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Make sure base_dirs is a list
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    # Validate sample_strategy
    if sample_strategy not in ['random', 'metadata_time', 'center']:
        print(f"Warning: Invalid sample strategy '{sample_strategy}'. Using 'random' instead.")
        sample_strategy = 'random'
    
    if sample_strategy == 'metadata_time' and time_column is None:
        print(f"Warning: 'metadata_time' strategy requires time_column. Using 'random' instead.")
        sample_strategy = 'random'
    
    # 1. Load ALL metadata
    df = pd.read_csv(metadata_csv)
    original_count = len(df)
    original_distribution = df['video_type'].value_counts()
    
    # 2. Check if split column exists
    if split_column not in df.columns:
        raise ValueError(f"Split column '{split_column}' not found in metadata. "
                        f"Available columns: {list(df.columns)}")
    
    # 3. Validate split values
    valid_splits = {'train', 'val', 'test'}
    actual_splits = set(df[split_column].dropna().str.lower())
    
    if not actual_splits.issubset(valid_splits):
        invalid_splits = actual_splits - valid_splits
        raise ValueError(f"Invalid split values found: {invalid_splits}. "
                        f"Valid values are: {valid_splits}")
    
    # Check if all required splits are present
    if validate_split:
        required_splits = {'train', 'val', 'test'}
        missing_splits = required_splits - actual_splits
        if missing_splits:
            raise ValueError(f"Missing required split(s): {missing_splits}. "
                           f"Found splits: {actual_splits}")
    
    # 4. Normalize split values to lowercase
    df[split_column] = df[split_column].str.lower()
    
    # 5. Remove rows with missing split values
    df_with_split = df.dropna(subset=[split_column]).copy()
    
    print(f"Original metadata: {original_count} rows")
    print(f"After removing missing splits: {len(df_with_split)} rows")
    
    # 6. Check which files exist across all directories
    existing_mask = []
    found_by_dir = {i: 0 for i in range(len(base_dirs))}
    
    for idx, row in tqdm(df_with_split.iterrows(), total=len(df_with_split), desc="Checking files"):
        video_id = row['id']
        found = False
        
        # Check each directory
        for dir_idx, base_dir in enumerate(base_dirs):
            video_dir = os.path.join(base_dir, video_id)
            if not os.path.exists(video_dir):
                continue
                
            files = os.listdir(video_dir)
            video_file = None
            
            for file in files:
                if file.endswith('.mp4') or file.endswith('.mov'):
                    video_file = os.path.join(video_dir, file)
                    break
                    
            sensor_file = os.path.join(base_dir, video_id, sensor_subdir, "Dashcam-Accelerometer_Acceleration.csv")
            
            if video_file and os.path.exists(video_file) and os.path.exists(sensor_file):
                found = True
                found_by_dir[dir_idx] += 1
                break
        
        existing_mask.append(found)
    
    # 7. Filter to only existing files
    existing_mask = np.array(existing_mask)
    existing_df = df_with_split[existing_mask].copy()
    existing_distribution = existing_df['video_type'].value_counts()
    
    print(f"After checking file existence: {len(existing_df)} rows")
    
    # 8. Split the data according to the split column
    train_df = existing_df[existing_df[split_column] == 'train'].copy()
    val_df = existing_df[existing_df[split_column] == 'val'].copy()
    test_df = existing_df[existing_df[split_column] == 'test'].copy()
    
    # 9. Validate that each split has samples
    if len(train_df) == 0:
        raise ValueError("No training samples found!")
    if len(val_df) == 0:
        print("Warning: No validation samples found!")
    if len(test_df) == 0:
        print("Warning: No test samples found!")
    
    # 10. Check class distribution in each split
    train_distribution = train_df['video_type'].value_counts()
    val_distribution = val_df['video_type'].value_counts()
    test_distribution = test_df['video_type'].value_counts()
    
    print(f"\nSplit sizes:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # 11. Check for missing classes in any split
    all_classes = set(existing_df['video_type'].unique())
    train_classes = set(train_df['video_type'].unique()) if len(train_df) > 0 else set()
    val_classes = set(val_df['video_type'].unique()) if len(val_df) > 0 else set()
    test_classes = set(test_df['video_type'].unique()) if len(test_df) > 0 else set()
    
    missing_in_train = all_classes - train_classes
    missing_in_val = all_classes - val_classes
    missing_in_test = all_classes - test_classes
    
    if missing_in_train:
        print(f"Warning: Classes missing in training set: {missing_in_train}")
    if missing_in_val and len(val_df) > 0:
        print(f"Warning: Classes missing in validation set: {missing_in_val}")
    if missing_in_test and len(test_df) > 0:
        print(f"Warning: Classes missing in test set: {missing_in_test}")
    
    # 12. Create datasets
    train_data = NvidiaDashcamDataset(
        train_df, 
        base_dirs, 
        is_train=True, 
        skip_missing=True, 
        transform=create_video_transforms(
            mode='train',
            enable_custom_augmentation=True,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            saturation_range=(0.9, 1.1),
            rotation_range=(-5, 5),
        ),
        sensor_subdir=sensor_subdir,
        sample_strategy=sample_strategy,
        time_column=time_column
    )
    
    val_data = None
    if len(val_df) > 0:
        val_data = NvidiaDashcamDataset(
            val_df, 
            base_dirs, 
            is_train=False, 
            skip_missing=True, 
            transform=create_video_transforms(mode='val'),
            sensor_subdir=sensor_subdir,
            sample_strategy=sample_strategy,
            time_column=time_column
        )
    
    test_data = None
    if len(test_df) > 0:
        test_data = NvidiaDashcamDataset(
            test_df, 
            base_dirs, 
            is_train=False, 
            skip_missing=True, 
            transform=create_video_transforms(mode='val'),
            sensor_subdir=sensor_subdir,
            sample_strategy=sample_strategy,
            time_column=time_column
        )
    
    # 13. Display visual statistics if requested
    if show_stats:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display, HTML
        
        # Set the style
        plt.style.use('fivethirtyeight')
        sns.set_palette('colorblind')
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("NVIDIA Dashcam Dataset Statistics (Manual Split)", fontsize=20, fontweight='bold')
        
        # 1. Distribution comparison - Original vs. Filtered
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
        
        all_classes = sorted(set(original_distribution.index) | set(existing_distribution.index))
        x = np.arange(len(all_classes))
        width = 0.35
        
        # Get data for each distribution, filling in zeros for missing classes
        orig_data = [original_distribution.get(cls, 0) for cls in all_classes]
        exist_data = [existing_distribution.get(cls, 0) for cls in all_classes]
        
        # Plot bars
        ax1.bar(x - width/2, orig_data, width, label='Original')
        ax1.bar(x + width/2, exist_data, width, label='Found Files')
        
        # Add labels and legend
        ax1.set_title('Class Distribution: Original vs. Found Files')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_classes, rotation=45, ha='right')
        ax1.legend()
        
        # Add counts on bars
        for i, v in enumerate(orig_data):
            if v > 0:
                ax1.text(i - width/2, v + max(orig_data)*0.01, f"{v}", ha='center', fontsize=9)
        for i, v in enumerate(exist_data):
            if v > 0:
                ax1.text(i + width/2, v + max(exist_data)*0.01, f"{v}", ha='center', fontsize=9)
        
        # 2. Split distribution pie chart
        ax2 = plt.subplot2grid((2, 3), (0, 2))
        split_counts = [len(train_df), len(val_df), len(test_df)]
        split_labels = ['Train', 'Validation', 'Test']
        # Remove empty splits from pie chart
        non_zero_splits = [(label, count) for label, count in zip(split_labels, split_counts) if count > 0]
        if non_zero_splits:
            labels, counts = zip(*non_zero_splits)
            ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        ax2.set_title('Train/Val/Test Split')
        
        # 3. Train/Val/Test Split Distribution by Class
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        
        # Get data for splits that exist
        available_splits = []
        available_distributions = []
        available_dfs = []
        
        if len(train_df) > 0:
            available_splits.append("Train")
            available_distributions.append(train_distribution)
            available_dfs.append(train_df)
        if len(val_df) > 0:
            available_splits.append("Validation")
            available_distributions.append(val_distribution)
            available_dfs.append(val_df)
        if len(test_df) > 0:
            available_splits.append("Test")
            available_distributions.append(test_distribution)
            available_dfs.append(test_df)
        
        # Create a bar for each class in each split
        if available_splits:
            for i, cls in enumerate(existing_distribution.index):
                class_data = [dist.get(cls, 0) for dist in available_distributions]
                offset = (i - len(existing_distribution.index)/2 + 0.5) * (width * 0.8)
                ax3.bar(np.arange(len(available_splits)) + offset, class_data, width * 0.8, label=cls)
            
            # Add labels and legend
            ax3.set_title('Train/Val/Test Split by Class')
            ax3.set_xticks(np.arange(len(available_splits)))
            ax3.set_xticklabels(available_splits)
            ax3.legend(loc='upper right')
            
            # Add percentage annotations
            for i, split_df in enumerate(available_dfs):
                total = len(split_df)
                for cls in existing_distribution.index:
                    count = split_df['video_type'].value_counts().get(cls, 0)
                    if count > 0 and total > 0:
                        percentage = count / total * 100
                        # Find the class position
                        cls_idx = list(existing_distribution.index).index(cls)
                        bar_x = i + (cls_idx - len(existing_distribution.index)/2 + 0.5) * (width * 0.8)
                        ax3.text(bar_x, count + max(count, 1)*0.05, f"{percentage:.1f}%", 
                               ha='center', fontsize=8, rotation=90)
        
        # Create a text box with summary statistics
        stats_text = f"""
        <div style="font-family: Arial; background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin: 20px;">
            <h2 style="color: #333; text-align: center;">Dataset Summary (Manual Split)</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #ddd;">
                    <th style="text-align: left; padding: 8px;">Metric</th>
                    <th style="text-align: right; padding: 8px;">Value</th>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Original records</td>
                    <td style="text-align: right; padding: 8px;">{original_count}</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Records with split info</td>
                    <td style="text-align: right; padding: 8px;">{len(df_with_split)} ({len(df_with_split)/original_count*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Valid records found</td>
                    <td style="text-align: right; padding: 8px;">{len(existing_df)} ({len(existing_df)/len(df_with_split)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Training set size</td>
                    <td style="text-align: right; padding: 8px;">{len(train_df)} ({len(train_df)/len(existing_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Validation set size</td>
                    <td style="text-align: right; padding: 8px;">{len(val_df)} ({len(val_df)/len(existing_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Test set size</td>
                    <td style="text-align: right; padding: 8px;">{len(test_df)} ({len(test_df)/len(existing_df)*100:.1f}%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Sampling strategy</td>
                    <td style="text-align: right; padding: 8px;">{sample_strategy}</td>
                </tr>
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">Split method</td>
                    <td style="text-align: right; padding: 8px;">Manual (from {split_column} column)</td>
                </tr>
            </table>
        </div>
        """
        
        display(HTML(stats_text))
        
        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # Display class distribution details
        class_stats_html = f"""
        <div style="font-family: Arial; background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin: 20px;">
            <h2 style="color: #333; text-align: center;">Class Distribution Details</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #ddd; background-color: #f2f2f2;">
                    <th style="text-align: left; padding: 8px;">Class</th>
                    <th style="text-align: right; padding: 8px;">Original</th>
                    <th style="text-align: right; padding: 8px;">Found</th>
                    <th style="text-align: right; padding: 8px;">Training</th>
                    <th style="text-align: right; padding: 8px;">Validation</th>
                    <th style="text-align: right; padding: 8px;">Test</th>
                </tr>
        """
        
        all_classes_final = sorted(set(original_distribution.index) | set(existing_distribution.index))
        for cls in all_classes_final:
            orig = original_distribution.get(cls, 0)
            exist = existing_distribution.get(cls, 0)
            train = train_distribution.get(cls, 0) if len(train_df) > 0 else 0
            val = val_distribution.get(cls, 0) if len(val_df) > 0 else 0
            test = test_distribution.get(cls, 0) if len(test_df) > 0 else 0
            
            # Calculate retention percentage
            retention = (exist / orig * 100) if orig > 0 else 0
            
            # Color code based on retention
            color = "#d4edda"  # Green for good retention
            if retention < 90:
                color = "#fff3cd"  # Yellow for moderate retention
            if retention < 70:
                color = "#f8d7da"  # Red for poor retention
                
            class_stats_html += f"""
                <tr style="border-bottom: 1px solid #ddd; background-color: {color};">
                    <td style="padding: 8px;"><b>{cls}</b></td>
                    <td style="text-align: right; padding: 8px;">{orig}</td>
                    <td style="text-align: right; padding: 8px;">{exist} ({retention:.1f}%)</td>
                    <td style="text-align: right; padding: 8px;">{train} ({train/exist*100:.1f}% of found)</td>
                    <td style="text-align: right; padding: 8px;">{val} ({val/exist*100:.1f}% of found)</td>
                    <td style="text-align: right; padding: 8px;">{test} ({test/exist*100:.1f}% of found)</td>
                </tr>
            """
        
        class_stats_html += """
            </table>
        </div>
        """
        
        display(HTML(class_stats_html))
    
    return train_data, val_data, test_data


def add_split_column_to_metadata(metadata_csv, output_csv=None, train_ratio=0.7, val_ratio=0.15, 
                                test_ratio=0.15, seed=42, stratify=True):
    """
    Add a split column to existing metadata CSV file
    
    Args:
        metadata_csv: Path to existing metadata CSV
        output_csv: Path to save updated CSV (if None, overwrites original)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        stratify: Whether to stratify by video_type
        
    Returns:
        Updated DataFrame with split column
    """
    from sklearn.model_selection import train_test_split
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio}")
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    
    # Check if split column already exists
    if 'split' in df.columns:
        print("Warning: 'split' column already exists. Overwriting...")
    
    # Set random seed
    np.random.seed(seed)
    
    if stratify and 'video_type' in df.columns:
        # Stratified split
        print("Using stratified split based on video_type")
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio), 
            random_state=seed, 
            stratify=df['video_type']
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=(test_ratio / (val_ratio + test_ratio)), 
                random_state=seed, 
                stratify=temp_df['video_type']
            )
        elif val_ratio > 0:
            val_df = temp_df
            test_df = pd.DataFrame()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
    else:
        # Random split
        print("Using random split (no stratification)")
        
        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calculate split indices
        n_total = len(df_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split the data
        train_df = df_shuffled[:n_train]
        val_df = df_shuffled[n_train:n_train + n_val]
        test_df = df_shuffled[n_train + n_val:]
    
    # Initialize split column
    df['split'] = ''
    
    # Assign split values
    df.loc[train_df.index, 'split'] = 'train'
    if len(val_df) > 0:
        df.loc[val_df.index, 'split'] = 'val'
    if len(test_df) > 0:
        df.loc[test_df.index, 'split'] = 'test'
    
    # Print split statistics
    split_counts = df['split'].value_counts()
    print(f"\nSplit distribution:")
    for split_type in ['train', 'val', 'test']:
        count = split_counts.get(split_type, 0)
        percentage = (count / len(df)) * 100
        print(f"  {split_type}: {count} ({percentage:.1f}%)")
    
    # Show class distribution by split if video_type exists
    if 'video_type' in df.columns:
        print(f"\nClass distribution by split:")
        split_class_table = pd.crosstab(df['video_type'], df['split'], margins=True)
        print(split_class_table)
    
    # Save the updated metadata
    if output_csv is None:
        output_csv = metadata_csv
    
    df.to_csv(output_csv, index=False)
    print(f"\nUpdated metadata saved to: {output_csv}")
    
    return df