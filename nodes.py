import math
from enum import Enum

import torch

# Try to import from ComfyUI-LTXVideo if available
try:
    from ComfyUI_LTXVideo.nodes_registry import comfy_node
    LTXVIDEO_AVAILABLE = True
except ImportError:
    # Fallback to standard ComfyUI node registration
    try:
        import comfy.extras
        LTXVIDEO_AVAILABLE = False
    except ImportError:
        LTXVIDEO_AVAILABLE = False


class Modality(Enum):
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class FrequencyDecompositionMethod(Enum):
    LAPLACIAN = "laplacian"
    WAVELET = "wavelet"


# Simple comfy_node decorator for standalone use
if not LTXVIDEO_AVAILABLE:
    class comfy_node:
        def __init__(self, name=None, category=None, **kwargs):
            self.name = name
            self.category = category
            self.kwargs = kwargs

        def __call__(self, cls):
            cls.NODE_NAME = self.name
            if self.category:
                cls.CATEGORY = self.category
            cls.COMFY_NODE_KWARGS = self.kwargs
            return cls


class FDGParameters:
    """
    Frequency-Decoupled Guidance parameters.

    Extends the standard CFG with separate guidance scales for
    low and high frequency components.

    Based on Algorithm 1 from the paper:
    - Low-frequency guidance: Controls global structure and condition alignment
    - High-frequency guidance: Enhances visual fidelity and details

    Recommended values (from Table 8 in paper):
    - w_low: 1.0-3.0 (conservative to avoid oversaturation)
    - w_high: 2.0-12.0 (stronger for better details)
    """

    def __init__(
        self,
        cfg_scale: float = 1.0,
        # FDG-specific parameters
        fdg_enabled: bool = True,
        w_low: float = 1.5,
        w_high: float = 4.0,
        frequency_levels: int = 2,
        decomposition_method: str = "laplacian",
        # Optional APG projection for better colors
        use_projection: bool = False,
        projection_weight: float = 1.0,
        # Standard parameters (inherited from GuiderParameters)
        stg_scale: float = 0.0,
        perturb_attn: bool = True,
        rescale_scale: float = 0.0,
        modality_scale: float = 1.0,
        skip_step: int = 0,
        cross_attn: bool = True,
    ):
        self.cfg_scale = cfg_scale
        self.stg_scale = stg_scale
        self.perturb_attn = perturb_attn
        self.rescale_scale = rescale_scale
        self.modality_scale = modality_scale
        self.skip_step = skip_step
        self.cross_attn = cross_attn

        # FDG-specific
        self.fdg_enabled = fdg_enabled
        self.w_low = w_low
        self.w_high = w_high
        self.frequency_levels = frequency_levels
        self.decomposition_method = decomposition_method
        self.use_projection = use_projection
        self.projection_weight = projection_weight

    def __str__(self):
        base_str = (
            f"cfg_scale: {self.cfg_scale}, stg_scale: {self.stg_scale}, "
            f"rescale_scale: {self.rescale_scale}, modality_scale: {self.modality_scale}"
        )
        if self.fdg_enabled:
            fdg_str = (
                f", FDG(enabled, w_low={self.w_low}, w_high={self.w_high}, "
                f"levels={self.frequency_levels}, method={self.decomposition_method})"
            )
            if self.use_projection:
                fdg_str += f", projection_weight={self.projection_weight}"
            return base_str + fdg_str
        return base_str

    def __repr__(self):
        return self.__str__()

    def calculate(
        self, noise_pred_pos, noise_pred_neg, noise_pred_pertubed, noise_pred_modality
    ):
        """
        Calculate the guided noise prediction.

        This method will be called during sampling. If FDG is enabled,
        it applies frequency-decoupled guidance instead of standard CFG.
        """
        if not self.fdg_enabled:
            # Standard CFG calculation
            noise_pred = (
                noise_pred_pos
                + (self.cfg_scale - 1) * (noise_pred_pos - noise_pred_neg)
                + self.stg_scale * (noise_pred_pos - noise_pred_pertubed)
                + (self.modality_scale - 1) * (noise_pred_pos - noise_pred_modality)
            )
        else:
            # FDG calculation
            from .fdg_utils import apply_fdg_guidance, apply_fdg_with_projection

            # When cfg_scale is 1.0, noise_pred_neg is 0 (not computed)
            # In this case, just use the positive prediction directly
            if isinstance(noise_pred_neg, int) and noise_pred_neg == 0:
                cfg_component = noise_pred_pos
            elif self.use_projection:
                cfg_component = apply_fdg_with_projection(
                    pred_cond=noise_pred_pos,
                    pred_uncond=noise_pred_neg,
                    w_low=self.w_low,
                    w_high=self.w_high,
                    parallel_weight=self.projection_weight,
                    levels=self.frequency_levels,
                )
            else:
                cfg_component = apply_fdg_guidance(
                    pred_cond=noise_pred_pos,
                    pred_uncond=noise_pred_neg,
                    w_low=self.w_low,
                    w_high=self.w_high,
                    levels=self.frequency_levels,
                )

            # Combine with other guidance methods (STG, modality)
            noise_pred = (
                cfg_component
                + self.stg_scale * (noise_pred_pos - noise_pred_pertubed)
                + (self.modality_scale - 1) * (noise_pred_pos - noise_pred_modality)
            )

        # Apply rescaling if enabled
        if self.rescale_scale != 0:
            factor = noise_pred_pos.std() / noise_pred.std()
            factor = self.rescale_scale * factor + (1 - self.rescale_scale)
            noise_pred = noise_pred * factor

        return noise_pred

    def do_uncond(self):
        return not math.isclose(self.cfg_scale, 1.0)

    def do_perturbed(self):
        return not math.isclose(self.stg_scale, 0.0)

    def do_modality(self):
        return not math.isclose(self.modality_scale, 1.0)

    def do_skip(self, step: int) -> bool:
        if self.skip_step == 0:
            return False
        return step % (self.skip_step + 1) != 0

    def do_cross_attn(self, step: int) -> bool:
        return self.cross_attn and not self.do_skip(step)


@comfy_node(name="FDGParameters")
class FDGParametersNode:
    """
    ComfyUI node for configuring Frequency-Decoupled Guidance parameters.

    This node allows you to configure FDG for your LTXV video generation.
    Based on the paper "Guidance in the Frequency Domain Enables
    High-Fidelity Sampling at Low CFG Scales".

    Recommended settings from the paper:
    - For high quality at low CFG: w_low=1.0-1.5, w_high=3.0-7.0
    - For best quality: w_low=3.0-5.0, w_high=7.0-12.0
    - frequency_levels: 2 (default) provides good balance

    Examples from Table 8:
    - Stable Diffusion XL: w_low=3.0, w_high=10.0 (with cfg=3)
    - Stable Diffusion 3: w_low=1.5, w_high=12.0 (with cfg=1.5)
    - EDM2-XL: w_low=1.0, w_high=2.0 (with cfg=2)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modality": (
                    [m.value for m in Modality],
                    {"default": Modality.VIDEO.value},
                ),
                # CFG base scale
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Base CFG scale (used when FDG is disabled or for other guidance types)",
                    },
                ),
                # FDG enable
                "fdg_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable Frequency-Decoupled Guidance",
                    },
                ),
                # FDG low-frequency guidance scale
                "w_low": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "Low-frequency guidance scale (controls global structure). Lower values = better diversity, higher values = better condition alignment",
                    },
                ),
                # FDG high-frequency guidance scale
                "w_high": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "High-frequency guidance scale (controls details). Higher values = sharper details",
                    },
                ),
                # Frequency decomposition settings
                "frequency_levels": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Number of frequency decomposition levels. 2 is recommended",
                    },
                ),
                "decomposition_method": (
                    [m.value for m in FrequencyDecompositionMethod],
                    {
                        "default": FrequencyDecompositionMethod.LAPLACIAN.value,
                        "tooltip": "Method for frequency decomposition",
                    },
                ),
                # Optional APG projection
                "use_projection": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use APG-style orthogonal projection for better color composition",
                    },
                ),
                "projection_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Weight for parallel component in APG projection",
                    },
                ),
                # Other guidance parameters
                "stg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "perturb_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "modality_scale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "skip_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "cross_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "parameters": (
                    "GUIDER_PARAMETERS",
                    {"default": None},
                ),
            },
        }

    RETURN_TYPES = ("GUIDER_PARAMETERS",)

    FUNCTION = "get_parameters"
    CATEGORY = "lightricks/LTXV"

    def get_parameters(
        self,
        modality,
        cfg,
        fdg_enabled,
        w_low,
        w_high,
        frequency_levels,
        decomposition_method,
        use_projection,
        projection_weight,
        stg,
        perturb_attn,
        rescale,
        modality_scale,
        skip_step,
        cross_attn,
        parameters=None,
    ):
        parameters = parameters.copy() if parameters is not None else {}

        if modality in parameters:
            raise ValueError(f"Modality {modality} already exists in parameters")

        parameters.update(
            {
                modality: FDGParameters(
                    cfg_scale=cfg,
                    fdg_enabled=fdg_enabled,
                    w_low=w_low,
                    w_high=w_high,
                    frequency_levels=frequency_levels,
                    decomposition_method=decomposition_method,
                    use_projection=use_projection,
                    projection_weight=projection_weight,
                    stg_scale=stg,
                    perturb_attn=perturb_attn,
                    rescale_scale=rescale,
                    modality_scale=modality_scale,
                    skip_step=skip_step,
                    cross_attn=cross_attn,
                )
            }
        )

        return (parameters,)
