"""
Depth Anything 3 package (relocated under projects.mmdet3d_plugin.models).
This module aliases itself to `depth_anything_3` so legacy absolute imports
continue to work without a top-level shim.
"""

import sys

# Alias the current package to the legacy name for backward-compat imports
sys.modules.setdefault("depth_anything_3", sys.modules[__name__])

# Ensure cfg is imported and aliased so `depth_anything_3.cfg` works
from . import cfg as _da3_cfg  # noqa: E402
sys.modules["depth_anything_3.cfg"] = _da3_cfg


