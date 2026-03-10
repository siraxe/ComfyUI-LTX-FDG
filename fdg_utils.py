import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def build_laplacian_pyramid(
    tensor: torch.Tensor,
    levels: int = 2,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> List[torch.Tensor]:
    """
    Build a Laplacian pyramid from a tensor.

    Args:
        tensor: Input tensor of shape [B, C, H, W] or [B, C, T, H, W] for video
        levels: Number of levels in the pyramid
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel

    Returns:
        List of tensors representing the Laplacian pyramid levels
    """
    pyramid = []
    current = tensor

    # Helper function to process 4D tensors
    def process_4d_level(input_4d, orig_h, orig_w):
        """Apply Gaussian blur, downsample, and upsample a 4D tensor."""
        # Blur
        blurred_4d = _gaussian_blur_2d(input_4d, kernel_size, sigma)
        # Downsample
        downsampled_4d = F.interpolate(
            blurred_4d, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        # Upsample back to original size
        upsampled_4d = F.interpolate(
            downsampled_4d, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        return downsampled_4d, upsampled_4d

    for i in range(levels - 1):
        H, W = current.shape[-2:]

        if current.dim() == 5:  # Video [B, C, T, H, W]
            B, C, T = current.shape[:3]
            # Reshape to [B*T, C, H, W] for spatial operations
            current_2d = current.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

            # Process in 2D
            downsampled_2d, upsampled_2d = process_4d_level(current_2d, H, W)

            # Compute Laplacian in 2D
            laplacian_2d = current_2d - upsampled_2d

            # Reshape back to [B, C, T, H, W]
            laplacian = laplacian_2d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

            # Reshape downsampled back to [B, C, T, H, W] for next iteration
            current = downsampled_2d.reshape(B, T, C, downsampled_2d.shape[2], downsampled_2d.shape[3]).permute(0, 2, 1, 3, 4)
        else:  # Image [B, C, H, W]
            # Process in 2D directly
            downsampled, upsampled = process_4d_level(current, H, W)

            # Compute Laplacian
            laplacian = current - upsampled

            # Continue with downsampled version
            current = downsampled

        pyramid.append(laplacian)

    # Add the final low-frequency component
    pyramid.append(current)

    return pyramid


def gaussian_blur(
    tensor: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Gaussian blur to a tensor.

    Args:
        tensor: Input tensor of shape [B, C, H, W] or [B, C, T, H, W]
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Blurred tensor
    """
    if tensor.dim() == 5:  # Video [B, C, T, H, W]
        B, C, T, H, W = tensor.shape
        # Reshape to [B*T, C, H, W] for 2D convolution
        tensor_2d = tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        blurred = _gaussian_blur_2d(tensor_2d, kernel_size, sigma)
        # Reshape back
        blurred = blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    else:  # Image [B, C, H, W]
        blurred = _gaussian_blur_2d(tensor, kernel_size, sigma)

    return blurred


def _gaussian_blur_2d(
    tensor: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply 2D Gaussian blur."""
    # Create Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()

    # Reshape for conv2d: [out_channels, in_channels, H, W]
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(tensor.shape[1], 1, 1, 1).to(tensor.device)

    # Apply separable convolution for efficiency
    padding = kernel_size // 2
    tensor = F.pad(tensor, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(tensor, kernel, groups=tensor.shape[1])

    return blurred


def build_image_from_pyramid(pyramid: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct an image from a Laplacian pyramid.

    Args:
        pyramid: List of tensors representing the Laplacian pyramid levels

    Returns:
        Reconstructed tensor
    """
    # Start with the lowest frequency component
    result = pyramid[-1]

    # Check if we're dealing with video (5D) or image (4D)
    is_video = result.dim() == 5

    # Work backwards through the pyramid
    for i in range(len(pyramid) - 2, -1, -1):
        target_size = pyramid[i].shape[-2:]

        if is_video:
            # For video, reshape to 4D for interpolation
            B, C, T, H, W = result.shape
            result_2d = result.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            upsampled_2d = F.interpolate(
                result_2d, size=target_size, mode='bilinear', align_corners=False
            )
            result = upsampled_2d.reshape(B, T, C, target_size[0], target_size[1]).permute(0, 2, 1, 3, 4)
        else:
            # For image, interpolate directly
            result = F.interpolate(
                result, size=target_size, mode='bilinear', align_corners=False
            )

        # Add the Laplacian detail
        result = result + pyramid[i]

    return result


def apply_fdg_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    w_low: float = 1.5,
    w_high: float = 4.0,
    levels: int = 2,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply Frequency-Decoupled Guidance to model predictions.

    This implements the FDG algorithm from the paper:
    - Decompose predictions into frequency bands
    - Apply separate guidance scales to low and high frequencies
    - Reconstruct the guided prediction

    Args:
        pred_cond: Conditional prediction [B, C, ...]
        pred_uncond: Unconditional prediction [B, C, ...]
        w_low: Guidance scale for low-frequency components
        w_high: Guidance scale for high-frequency components
        levels: Number of frequency levels
        kernel_size: Gaussian kernel size for pyramid construction
        sigma: Gaussian sigma for pyramid construction

    Returns:
        Guided prediction with frequency-decoupled CFG
    """
    # Build Laplacian pyramids for both predictions
    pyramid_cond = build_laplacian_pyramid(pred_cond, levels, kernel_size, sigma)
    pyramid_uncond = build_laplacian_pyramid(pred_uncond, levels, kernel_size, sigma)

    guided_pyramid = []

    # Apply different guidance scales to different frequency levels
    for idx, (p_cond, p_uncond) in enumerate(zip(pyramid_cond, pyramid_uncond)):
        # Last level is the lowest frequency
        if idx == len(pyramid_cond) - 1:
            # Low-frequency component - use conservative guidance
            scale = w_low
        else:
            # High-frequency components - use stronger guidance
            scale = w_high

        # Apply CFG formula: D_u + w * (D_c - D_u)
        p_guided = p_uncond + scale * (p_cond - p_uncond)
        guided_pyramid.append(p_guided)

    # Reconstruct from guided pyramid
    pred_guided = build_image_from_pyramid(guided_pyramid)

    return pred_guided


def apply_fdg_with_projection(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    w_low: float = 1.5,
    w_high: float = 4.0,
    parallel_weight: float = 1.0,
    levels: int = 2,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply FDG with optional APG-style orthogonal projection.

    This combines FDG with Adaptive Projected Guidance for better
    color composition and fewer artifacts.

    Args:
        pred_cond: Conditional prediction [B, C, ...]
        pred_uncond: Unconditional prediction [B, C, ...]
        w_low: Guidance scale for low-frequency components
        w_high: Guidance scale for high-frequency components
        parallel_weight: Weight for parallel component (1.0 = full projection)
        levels: Number of frequency levels
        kernel_size: Gaussian kernel size for pyramid construction
        sigma: Gaussian sigma for pyramid construction

    Returns:
        Guided prediction with FDG + APG
    """
    # Build pyramids
    pyramid_cond = build_laplacian_pyramid(pred_cond, levels, kernel_size, sigma)
    pyramid_uncond = build_laplacian_pyramid(pred_uncond, levels, kernel_size, sigma)

    guided_pyramid = []

    for idx, (p_cond, p_uncond) in enumerate(zip(pyramid_cond, pyramid_uncond)):
        # Compute the difference (guidance direction)
        diff = p_cond - p_uncond

        # Project onto conditional prediction (APG-style)
        diff_parallel, diff_orthogonal = project_orthogonal(diff, p_cond)

        # Combine with weights
        diff = parallel_weight * diff_parallel + diff_orthogonal

        # Apply guidance scale
        scale = w_low if idx == len(pyramid_cond) - 1 else w_high
        p_guided = p_uncond + scale * diff

        guided_pyramid.append(p_guided)

    # Reconstruct
    pred_guided = build_image_from_pyramid(guided_pyramid)

    return pred_guided


def project_orthogonal(
    v0: torch.Tensor,
    v1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project v0 onto v1 and get orthogonal component.

    Args:
        v0: Vector to project
        v1: Vector to project onto

    Returns:
        Tuple of (parallel_component, orthogonal_component)
    """
    original_dtype = v0.dtype

    # Use double precision for numerical stability
    if v0.dtype != torch.float64:
        v0 = v0.double()
    if v1.dtype != torch.float64:
        v1 = v1.double()

    # Normalize v1
    v1_normalized = F.normalize(v1, dim=[-1, -2, -3] if v1.dim() >= 3 else [-1, -2])

    # Project v0 onto v1
    v0_parallel = (v0 * v1_normalized).sum(dim=list(range(-v0.dim(), 0)), keepdim=True) * v1_normalized
    v0_orthogonal = v0 - v0_parallel

    return v0_parallel.to(original_dtype), v0_orthogonal.to(original_dtype)


# Frequency decomposition using Discrete Wavelet Transform (alternative to Laplacian)
def apply_dwt_frequency_decomposition(
    tensor: torch.Tensor,
    levels: int = 1,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Apply Discrete Wavelet Transform for frequency decomposition.

    This is an alternative to Laplacian pyramids, as mentioned in the paper.
    The DWT provides better localization in frequency domain.

    Args:
        tensor: Input tensor [B, C, H, W] or [B, C, T, H, W]
        levels: Number of decomposition levels

    Returns:
        Tuple of (low_frequency, list_of_high_frequencies)
    """
    # Note: This is a simplified implementation
    # For production, consider using pywt or kornia for proper DWT
    low_freq = tensor
    high_freqs = []

    for _ in range(levels):
        # Simple approximation: high pass via subtracting low pass
        low_freq_blurred = gaussian_blur(low_freq, kernel_size=5, sigma=1.0)
        high_freq = low_freq - low_freq_blurred
        high_freqs.append(high_freq)

        # Downsample for next level
        if low_freq.dim() == 5:  # Video
            low_freq = F.interpolate(
                low_freq_blurred, scale_factor=0.5, mode='bilinear', align_corners=False
            )
        else:  # Image
            low_freq = F.interpolate(
                low_freq_blurred, scale_factor=0.5, mode='bilinear', align_corners=False
            )

    return low_freq, high_freqs
