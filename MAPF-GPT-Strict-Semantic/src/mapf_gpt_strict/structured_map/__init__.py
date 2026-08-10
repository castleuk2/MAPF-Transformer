"""Vendored unchanged from ../mapf-structured-map-transformer.

The source modules are kept byte-for-byte identical so patch construction and
the 17x17 halo -> 25 latent-token path cannot silently diverge.
"""

from .config import ModelConfig
from .geometry import PatchGeometry, build_patch_geometry, patchify_core
from .model import MapEncoderOutput, StructuredMapTransformer

__all__ = [
    "MapEncoderOutput", "ModelConfig", "PatchGeometry",
    "StructuredMapTransformer", "build_patch_geometry", "patchify_core",
]
