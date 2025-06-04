import sys
import types
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import random
from typing import Dict, Union, Callable, List, Any, Optional, Tuple

# Fix for PyTorchVideo compatibility
from torchvision.transforms import functional as F_tv
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = F_tv.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

# Import PyTorchVideo transforms
from pytorchvideo.transforms import create_video_transform as ptv_create_video_transform

class VideoAugmentation(nn.Module):
    """
    Advanced video augmentation module with consistent frame transformations.
    Applies the same augmentation parameters to all frames in a video to preserve temporal consistency.
    All augmentations are applied before normalization for more natural-looking results.
    """
    def __init__(
        self,
        # Color transformations
        brightness_range: Tuple[float, float] = (1.0, 1.0),  # Brightness factor range
        contrast_range: Tuple[float, float] = (1.0, 1.0),    # Contrast factor range  
        saturation_range: Tuple[float, float] = (1.0, 1.0),  # Saturation factor range
        hue_range: Tuple[float, float] = (0.0, 0.0),         # Hue shift range (-0.5 to 0.5)
        
        # Geometric transformations
        rotation_range: Tuple[float, float] = (0.0, 0.0),    # Rotation in degrees
        scale_range: Tuple[float, float] = (1.0, 1.0),       # Scale factor range
        shear_range: Tuple[float, float] = (0.0, 0.0),       # Shear angle in degrees
        translate_range: Tuple[float, float] = (0.0, 0.0),   # Translation as fraction of image size
        
        # Special effects
        grayscale_prob: float = 0.0,                         # Probability of grayscale conversion
        noise_level: float = 0.0,                            # Gaussian noise level (0.0 = disabled)
        blur_sigma: float = 0.0,                             # Gaussian blur sigma (0.0 = disabled)
        
        # Spatial augmentations
        cutout_prob: float = 0.0,                            # Probability of applying cutout
        cutout_count_range: Tuple[int, int] = (1, 3),        # Range of number of cutouts
        cutout_size_range: Tuple[float, float] = (0.1, 0.2), # Size range as fraction of image
        
        # Advanced effects (use with caution)
        color_inversion_prob: float = 0.0,                   # Probability of color inversion
        solarization_prob: float = 0.0,                      # Probability of solarization
        solarization_threshold: float = 0.5,                 # Threshold for solarization
        posterization_prob: float = 0.0,                     # Probability of posterization
        posterization_bits_range: Tuple[int, int] = (3, 6),  # Range of bits for posterization
        
        # Global augmentation probability
        aug_probability: float = 1.0,                        # Overall probability of applying augmentations
        
        # Debug settings
        debug: bool = False                                  # Whether to print debug information
    ):
        """
        Initialize the video augmentation module with specified parameters.
        
        Parameters are designed to be intuitive with sensible defaults.
        All transformations will be applied consistently across frames.
        """
        super().__init__()
        
        # Store transformation parameters
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.translate_range = translate_range
        
        self.grayscale_prob = grayscale_prob
        self.noise_level = noise_level
        self.blur_sigma = blur_sigma
        
        self.cutout_prob = cutout_prob
        self.cutout_count_range = cutout_count_range
        self.cutout_size_range = cutout_size_range
        
        self.color_inversion_prob = color_inversion_prob
        self.solarization_prob = solarization_prob
        self.solarization_threshold = solarization_threshold
        self.posterization_prob = posterization_prob
        self.posterization_bits_range = posterization_bits_range
        
        self.aug_probability = aug_probability
        self.debug = debug

    def _sample_augmentation_parameters(self, shape: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Sample augmentation parameters once for the entire video.
        
        Args:
            shape: Video tensor shape (C, T, H, W)
            
        Returns:
            Dictionary of parameters to be applied to all frames
        """
        c, t, h, w = shape
        
        params = {}
        
        # Skip augmentation entirely with probability (1 - aug_probability)
        if random.random() > self.aug_probability:
            params['skip_augmentation'] = True
            return params
        
        # Color transformations
        params['brightness'] = random.uniform(self.brightness_range[0], self.brightness_range[1])
        params['contrast'] = random.uniform(self.contrast_range[0], self.contrast_range[1])
        params['saturation'] = random.uniform(self.saturation_range[0], self.saturation_range[1])
        params['hue'] = random.uniform(self.hue_range[0], self.hue_range[1])
        
        # Geometric transformations
        params['rotation'] = random.uniform(self.rotation_range[0], self.rotation_range[1])
        params['scale'] = random.uniform(self.scale_range[0], self.scale_range[1])
        params['shear'] = random.uniform(self.shear_range[0], self.shear_range[1])
        params['translate_x'] = random.uniform(-self.translate_range[1], self.translate_range[1]) * w
        params['translate_y'] = random.uniform(-self.translate_range[1], self.translate_range[1]) * h
        
        # Apply affine transformation if any geometric parameter is non-default
        params['apply_affine'] = (
            params['rotation'] != 0 or 
            params['scale'] != 1 or 
            params['shear'] != 0 or 
            params['translate_x'] != 0 or 
            params['translate_y'] != 0
        )
        
        # Special effects
        params['apply_grayscale'] = random.random() < self.grayscale_prob
        params['apply_noise'] = self.noise_level > 0
        params['apply_blur'] = self.blur_sigma > 0
        
        # Cutout
        params['apply_cutout'] = random.random() < self.cutout_prob
        if params['apply_cutout']:
            params['cutout_count'] = random.randint(
                self.cutout_count_range[0], 
                self.cutout_count_range[1]
            )
            params['cutout_boxes'] = []
            
            # Pre-generate cutout boxes for consistency
            for _ in range(params['cutout_count']):
                size_factor = random.uniform(
                    self.cutout_size_range[0], 
                    self.cutout_size_range[1]
                )
                
                cut_h = int(h * size_factor)
                cut_w = int(w * size_factor)
                
                # Ensure we don't go out of bounds
                max_top = max(0, h - cut_h - 1)
                max_left = max(0, w - cut_w - 1)
                
                if max_top > 0 and max_left > 0:  # Only add if valid
                    top = random.randint(0, max_top)
                    left = random.randint(0, max_left)
                    params['cutout_boxes'].append((top, left, cut_h, cut_w))
        
        # Advanced effects
        params['apply_color_inversion'] = random.random() < self.color_inversion_prob
        params['apply_solarization'] = random.random() < self.solarization_prob
        params['apply_posterization'] = random.random() < self.posterization_prob
        
        if params['apply_posterization']:
            params['posterization_bits'] = random.randint(
                self.posterization_bits_range[0],
                self.posterization_bits_range[1]
            )
        
        return params
    
    def _apply_cutout(self, frame: torch.Tensor, boxes: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """
        Apply cutout (random erasing) to a frame using pre-generated boxes.
        
        Args:
            frame: Frame tensor of shape [C, H, W]
            boxes: List of (top, left, height, width) tuples
            
        Returns:
            Frame with cutouts applied
        """
        for top, left, cut_h, cut_w in boxes:
            frame[:, top:top+cut_h, left:left+cut_w] = 0
        
        return frame
    
    def _apply_augmentations_to_frame(
        self, 
        frame: torch.Tensor, 
        params: Dict[str, Any],
        device: torch.device
    ) -> torch.Tensor:
        """
        Apply augmentations to a single frame using the parameters.
        
        Args:
            frame: Frame tensor of shape [C, H, W]
            params: Dictionary of parameters from _sample_augmentation_parameters
            device: Device to use for tensor operations
            
        Returns:
            Augmented frame tensor of shape [C, H, W]
        """
        # Skip augmentation if specified
        if params.get('skip_augmentation', False):
            return frame
        
        # Color transformations
        frame = F.adjust_brightness(frame, params['brightness'])
        frame = F.adjust_contrast(frame, params['contrast'])
        frame = F.adjust_saturation(frame, params['saturation'])
        frame = F.adjust_hue(frame, params['hue'])
        
        # Geometric transformations (affine)
        if params['apply_affine']:
            frame = F.affine(
                frame,
                angle=params['rotation'],
                scale=params['scale'],
                shear=params['shear'],
                translate=[params['translate_x'], params['translate_y']],
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0
            )
        
        # Grayscale
        if params['apply_grayscale']:
            frame = F.rgb_to_grayscale(frame, num_output_channels=3)
        
        # Noise
        if params['apply_noise']:
            noise = torch.randn_like(frame, device=device) * self.noise_level
            frame = torch.clamp(frame + noise, 0, 1)
        
        # Blur
        if params['apply_blur']:
            # Add batch dimension for gaussian_blur, then remove it
            frame = F.gaussian_blur(
                frame.unsqueeze(0),
                kernel_size=int(self.blur_sigma * 4) * 2 + 1,  # Must be odd
                sigma=self.blur_sigma
            ).squeeze(0)
        
        # Advanced effects
        if params['apply_posterization']:
            # Convert to byte tensor, posterize, then back to float
            frame_byte = (frame * 255).byte()
            frame_byte = F.posterize(frame_byte, params['posterization_bits'])
            frame = frame_byte.float() / 255.0
        
        if params['apply_solarization']:
            frame = F.solarize(frame, self.solarization_threshold)
        
        if params['apply_color_inversion']:
            frame = 1.0 - frame
        
        # Cutout
        if params['apply_cutout']:
            frame = self._apply_cutout(frame, params['cutout_boxes'])
        
        return frame
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply consistent augmentations to all frames in the video.
        
        Args:
            video: Video tensor of shape [C, T, H, W] with values in [0, 1]
            
        Returns:
            Augmented video tensor of shape [C, T, H, W]
        """
        c, t, h, w = video.shape
        device = video.device
        
        # Sample parameters once for the entire video
        params = self._sample_augmentation_parameters((c, t, h, w))
        
        # Print parameters if in debug mode
        if self.debug:
            print("Video Augmentation Parameters:")
            for k, v in params.items():
                # Skip printing cutout boxes as they're too verbose
                if k != 'cutout_boxes':
                    print(f"  {k}: {v}")
        
        # Process each frame with the same parameters
        augmented_frames = []
        for i in range(t):
            # Get current frame
            frame = video[:, i]
            
            # Apply augmentations
            frame = self._apply_augmentations_to_frame(frame, params, device)
            
            # Add to list
            augmented_frames.append(frame)
        
        # Stack frames back to video
        augmented_video = torch.stack(augmented_frames, dim=1)
        
        return augmented_video


def create_video_transform(
    mode: str = 'train',                      # 'train' or 'val'/'test'
    normalize: bool = True,                   # Whether to normalize with mean and std
    video_mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),  # Mean for normalization
    video_std: Tuple[float, float, float] = (0.225, 0.225, 0.225),  # Std for normalization
    
    # Basic resize/crop parameters
    min_size: int = 224,                      # Min size after resizing
    max_size: Optional[int] = None,           # Max size after resizing (for 'train' mode)
    crop_size: int = 224,                     # Final crop size
    center_crop: bool = False,                # If True, use center crop; if False, use random crop for 'train'
    
    # Basic augmentation parameters
    horizontal_flip_prob: float = 0.5,        # Probability of horizontal flip for 'train' mode
    
    # Advanced augmentation parameters (only used in 'train' mode)
    enable_advanced_augmentation: bool = False,  # Whether to use advanced augmentations
    
    # Color transformation ranges
    brightness_range: Tuple[float, float] = (0.8, 1.2),  # Brightness factor range
    contrast_range: Tuple[float, float] = (0.8, 1.2),    # Contrast factor range
    saturation_range: Tuple[float, float] = (0.8, 1.2),  # Saturation factor range
    hue_range: Tuple[float, float] = (-0.1, 0.1),        # Hue shift range (-0.5 to 0.5)
    
    # Geometric transformation ranges
    rotation_range: Tuple[float, float] = (-10, 10),     # Rotation in degrees
    scale_range: Tuple[float, float] = (0.9, 1.1),       # Scale factor range
    shear_range: Tuple[float, float] = (-5, 5),          # Shear angle in degrees
    translate_range: Tuple[float, float] = (0.0, 0.1),   # Translation as fraction of image
    
    # Special effects
    grayscale_prob: float = 0.02,                        # Probability of grayscale conversion
    noise_level: float = 0.02,                           # Gaussian noise level (0.0 = disabled)
    blur_sigma: float = 0.0,                             # Gaussian blur sigma (0.0 = disabled)
    cutout_prob: float = 0.0,                            # Probability of cutout
    
    # Advanced effects
    color_inversion_prob: float = 0.0,                   # Probability of color inversion
    solarization_prob: float = 0.0,                      # Probability of solarization 
    posterization_prob: float = 0.0,                     # Probability of posterization
    
    # Debug mode
    debug: bool = False                                  # Whether to print parameters
) -> nn.Module:
    """
    Creates a comprehensive video transform pipeline with proper temporal consistency.
    
    Args:
        mode: 'train' for training mode with augmentations or 'val'/'test' for evaluation
        normalize: Whether to normalize with mean and std
        video_mean: Mean values for normalization (RGB)
        video_std: Standard deviation values for normalization (RGB)
        min_size: Minimum size after resize
        max_size: Maximum size after resize (only used in train mode)
        crop_size: Size after cropping
        center_crop: Whether to use center crop (True) or random crop (False)
        horizontal_flip_prob: Probability of horizontal flip in train mode
        enable_advanced_augmentation: Whether to use advanced augmentations
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        saturation_range: Range for saturation adjustment
        hue_range: Range for hue adjustment
        rotation_range: Range for rotation
        scale_range: Range for scaling
        shear_range: Range for shearing
        translate_range: Range for translation
        grayscale_prob: Probability of grayscale conversion
        noise_level: Level of Gaussian noise
        blur_sigma: Sigma for Gaussian blur
        cutout_prob: Probability of applying cutout
        color_inversion_prob: Probability of color inversion
        solarization_prob: Probability of solarization
        posterization_prob: Probability of posterization
        debug: Whether to print debug information
    
    Returns:
        Transform module that can be applied to videos
    """
    transforms = []
    
    # Resize
    if mode == 'train' and max_size is not None and max_size > min_size:
        # For train mode, resize with random size from min_size to max_size
        size = random.randint(min_size, max_size)
    else:
        # For val/test mode, resize to fixed min_size
        size = min_size
    
    # Define resize function
    def resize_tensor(video):
        """Resize video tensor preserving aspect ratio"""
        c, t, h, w = video.shape
        
        # Calculate new dimensions preserving aspect ratio
        if h > w:
            new_h, new_w = size * h // w, size
        else:
            new_h, new_w = size, size * w // h
        
        # Resize each frame
        resized_frames = []
        for i in range(t):
            frame = F.resize(video[:, i], [new_h, new_w], antialias=True)
            resized_frames.append(frame)
        
        # Stack back to video
        return torch.stack(resized_frames, dim=1)
    
    transforms.append(resize_tensor)
    
    # Crop
    def crop_tensor(video, use_letterbox=True):
        """
        Crop video tensor or use letterboxing
        
        Args:
            video: Video tensor of shape [C, T, H, W]
            use_letterbox: If True, use letterbox method (resize with padding) instead of cropping
        """
        c, t, h, w = video.shape
        
        if use_letterbox:
            # Letterbox approach - resize with padding to maintain aspect ratio
            # Calculate target size while maintaining aspect ratio
            scale = min(crop_size / h, crop_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Calculate padding
            pad_h = (crop_size - new_h) // 2
            pad_w = (crop_size - new_w) // 2
            
            # Apply resize and padding to all frames
            resized_frames = []
            for i in range(t):
                # Resize frame
                frame = F.resize(video[:, i], [new_h, new_w], antialias=True)
                
                # Create new frame with padding
                padded_frame = torch.zeros(c, crop_size, crop_size, device=video.device)
                padded_frame[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = frame
                
                resized_frames.append(padded_frame)
            
            # Stack back to video
            return torch.stack(resized_frames, dim=1)
        else:
            # Original crop behavior
            if center_crop or mode != 'train':
                # Center crop
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
            else:
                # Random crop for training
                top = random.randint(0, h - crop_size) if h > crop_size else 0
                left = random.randint(0, w - crop_size) if w > crop_size else 0
            
            # Apply crop to all frames
            cropped_frames = []
            for i in range(t):
                frame = F.crop(video[:, i], top, left, crop_size, crop_size)
                cropped_frames.append(frame)
            
            # Stack back to video
            return torch.stack(cropped_frames, dim=1)
    
    transforms.append(crop_tensor)
    
    # Horizontal flip for train mode
    if mode == 'train' and horizontal_flip_prob > 0:
        def horizontal_flip(video):
            """Apply horizontal flip to video with probability"""
            if random.random() < horizontal_flip_prob:
                c, t, h, w = video.shape
                flipped_frames = []
                for i in range(t):
                    frame = F.hflip(video[:, i])
                    flipped_frames.append(frame)
                return torch.stack(flipped_frames, dim=1)
            return video
        
        transforms.append(horizontal_flip)
    
    # Advanced augmentations for train mode
    if mode == 'train' and enable_advanced_augmentation:
        # Create VideoAugmentation with specified parameters
        video_aug = VideoAugmentation(
            # Color transformations
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue_range=hue_range,
            
            # Geometric transformations
            rotation_range=rotation_range,
            scale_range=scale_range,
            shear_range=shear_range,
            translate_range=translate_range,
            
            # Special effects
            grayscale_prob=grayscale_prob,
            noise_level=noise_level,
            blur_sigma=blur_sigma,
            cutout_prob=cutout_prob,
            
            # Advanced effects
            color_inversion_prob=color_inversion_prob,
            solarization_prob=solarization_prob,
            posterization_prob=posterization_prob,
            
            # Debug
            debug=debug
        )
        
        transforms.append(video_aug)
    
    # Normalization
    if normalize:
        def normalize_tensor(video):
            """Normalize video tensor with mean and std"""
            c, t, h, w = video.shape
            mean = torch.tensor(video_mean).view(c, 1, 1, 1).to(video.device)
            std = torch.tensor(video_std).view(c, 1, 1, 1).to(video.device)
            return (video - mean) / std
        
        transforms.append(normalize_tensor)
    
    # Combine all transforms
    class VideoTransform(nn.Module):
        def __init__(self, transforms):
            super().__init__()
            self.transforms = transforms
        
        def forward(self, video):
            # Ensure video is float and in range [0, 1]
            if video.dtype != torch.float32:
                video = video.float()
            
            if video.max() > 1.0:
                video = video / 255.0
            
            # Apply all transforms in sequence
            for transform in self.transforms:
                video = transform(video)
            
            return video
    
    return VideoTransform(transforms)


def visualize_video_augmentations(video_tensor, transform, num_frames=5, figsize=(15, 10)):
    """
    Visualize the effect of video augmentations.
    
    Args:
        video_tensor: Input video tensor of shape [C, T, H, W]
        transform: Transform to apply
        num_frames: Number of frames to display
        figsize: Figure size
        
    Returns:
        None (displays plot)
    """
    import matplotlib.pyplot as plt
    
    # Clone input to avoid modifying it
    input_tensor = video_tensor.clone()
    
    # Apply transform
    with torch.no_grad():
        output_tensor = transform(input_tensor)
    
    # Select frames to display
    c, t, h, w = video_tensor.shape
    frame_indices = torch.linspace(0, t-1, num_frames).long()
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=figsize)
    
    # Helper for denormalization
    def denormalize(tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean
    
    # Show original and transformed frames
    for i, idx in enumerate(frame_indices):
        # Original frame
        orig_frame = video_tensor[:, idx].cpu()
        if orig_frame.max() <= 1.0:
            orig_img = orig_frame.permute(1, 2, 0).numpy()
        else:
            orig_img = (orig_frame / 255.0).permute(1, 2, 0).numpy()
        
        # Transformed frame
        trans_frame = output_tensor[:, idx].cpu()
        
        # Check if normalization was applied
        if trans_frame.min() < 0 or trans_frame.max() > 1.0:
            # Denormalize
            trans_frame = denormalize(trans_frame)
        
        trans_img = trans_frame.permute(1, 2, 0).numpy()
        trans_img = np.clip(trans_img, 0, 1)
        
        # Display frames
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original Frame {idx}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(trans_img)
        axes[1, i].set_title(f"Augmented Frame {idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_video_transforms(
    # Basic parameters
    mode='train',
    video_key=None,
    num_samples=75,
    convert_to_float=True,
    crop_size=224,
    
    # Normalization
    normalize=True,
    video_mean=(0.45, 0.45, 0.45),
    video_std=(0.225, 0.225, 0.225),
    
    # Size parameters
    min_size=224,
    max_size=320,
    
    # Basic augmentations (built into PyTorchVideo)
    horizontal_flip_prob=0.5,
    
    # --- CUSTOM AUGMENTATION PARAMETERS ---
    # Whether to use custom augmentation at all
    enable_custom_augmentation=False,
    
    # Overall augmentation probability
    aug_probability=1.0,
    
    # Color transformations
    brightness_range=(0.9, 1.1),      # Min/max factor for brightness
    contrast_range=(0.9, 1.1),        # Min/max factor for contrast
    saturation_range=(0.9, 1.1),      # Min/max factor for saturation
    hue_range=(-0.05, 0.05),          # Min/max absolute hue shift (-0.5 to 0.5)
    
    # Geometric transformations
    rotation_range=(-5, 5),           # Min/max rotation in degrees
    scale_range=(0.95, 1.05),         # Min/max scale factor
    shear_range=(-2, 2),              # Min/max shear in degrees
    translate_range=(0.0, 0.05),      # Min/max translation as fraction of image
    perspective_distortion=0.0,       # Strength of perspective distortion (0.0 = none)
    
    # Noise and artifacts
    noise_level=0.0,                  # Standard deviation of Gaussian noise
    blur_sigma=0.0,                   # Gaussian blur sigma
    jpeg_quality=0,                   # JPEG compression quality (0=disabled, 1-100=quality)
    
    # Special effects
    grayscale_prob=0.0,               # Probability of grayscale conversion
    cutout_prob=0.0,                  # Probability of random cutouts
    cutout_count=(1, 3),              # Min/max number of cutouts
    cutout_size_range=(0.1, 0.2),     # Min/max relative size of cutouts
    
    # Extreme effects (use with caution)
    color_inversion_prob=0.0,         # Probability of color inversion
    solarization_prob=0.0,            # Probability of solarization
    posterization_prob=0.0,           # Probability of posterization
    posterization_bits_range=(3, 6),  # Range of bits for posterization
    solarization_threshold=0.5,       # Threshold for solarization
    
    # Debug mode
    debug=False
):
    """
    Creates a comprehensive video transform pipeline with precise control over all augmentations.
    Uses letterboxing (padding) to preserve aspect ratio instead of cropping.
    """
    # Create our own transformation sequence instead of relying on PyTorchVideo's transform
    transforms = []
    
    # Define a letterbox resize function that preserves aspect ratio with padding
    def letterbox_resize(video):
        """Resize video tensor preserving aspect ratio and adding padding"""
        c, t, h, w = video.shape
        
        # Target size (square)
        target_size = crop_size
        
        # Calculate scale to fit the longest dimension to target_size
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Calculate padding
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        # Process each frame with resize and padding
        letterboxed_frames = []
        for i in range(t):
            # Get frame
            frame = video[:, i]
            
            # Resize frame preserving aspect ratio
            frame = F.resize(frame, [new_h, new_w], antialias=True)
            
            # Create padded frame (initialize with zeros = black)
            padded_frame = torch.zeros(c, target_size, target_size, device=video.device)
            
            # Place resized frame in the center
            padded_frame[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = frame
            
            letterboxed_frames.append(padded_frame)
        
        # Stack back to video
        return torch.stack(letterboxed_frames, dim=1)
    
    # Add letterbox resize to transformations
    transforms.append(letterbox_resize)
    
    # Horizontal flip for train mode
    if mode == 'train' and horizontal_flip_prob > 0:
        def horizontal_flip(video):
            """Apply horizontal flip to video with probability"""
            if random.random() < horizontal_flip_prob:
                c, t, h, w = video.shape
                flipped_frames = []
                for i in range(t):
                    frame = F.hflip(video[:, i])
                    flipped_frames.append(frame)
                return torch.stack(flipped_frames, dim=1)
            return video
        
        transforms.append(horizontal_flip)
    
    # Advanced augmentations for train mode
    if mode == 'train' and enable_custom_augmentation:
        # Create VideoAugmentation with specified parameters
        video_aug = VideoAugmentation(
            # Color transformations
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue_range=hue_range,
            
            # Geometric transformations
            rotation_range=rotation_range,
            scale_range=scale_range,
            shear_range=shear_range,
            translate_range=translate_range,
            
            # Special effects
            grayscale_prob=grayscale_prob,
            noise_level=noise_level,
            blur_sigma=blur_sigma,
            cutout_prob=cutout_prob,
            
            # Advanced effects
            color_inversion_prob=color_inversion_prob,
            solarization_prob=solarization_prob,
            posterization_prob=posterization_prob,
            
            # Debug
            debug=debug
        )
        
        transforms.append(video_aug)
    
    # Normalization
    if normalize:
        def normalize_tensor(video):
            """Normalize video tensor with mean and std"""
            c, t, h, w = video.shape
            mean = torch.tensor(video_mean).view(c, 1, 1, 1).to(video.device)
            std = torch.tensor(video_std).view(c, 1, 1, 1).to(video.device)
            return (video - mean) / std
        
        transforms.append(normalize_tensor)
    
    # Combine all transforms
    class VideoTransform(nn.Module):
        def __init__(self, transforms):
            super().__init__()
            self.transforms = transforms
        
        def forward(self, video):
            # Ensure video is float and in range [0, 1]
            if video.dtype != torch.float32:
                video = video.float()
            
            if video.max() > 1.0:
                video = video / 255.0
            
            # Apply all transforms in sequence
            for transform in self.transforms:
                video = transform(video)
            
            return video
    
    return VideoTransform(transforms)