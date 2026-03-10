"""
Frequency-Decoupled Guidance (FDG) for ComfyUI.

Based on the paper:
"Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales"
https://arxiv.org/abs/2506.19713
"""

from .nodes import FDGParametersNode

NODE_CLASS_MAPPINGS = {
    "FDGParameters": FDGParametersNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FDGParameters": "LTXV FDG Parameters",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
