# tests/conftest.py
# ---------------------------------------------------------------------------
# Pin one-init C extensions and lazy-registered plugins in sys.modules before
# pytest-cov's SysModuleSaver snapshots sys.modules. Any module evicted and
# re-imported in the same process fails:
#   - numpy 2.x: "cannot load module more than once per process"
#   - Pillow PNG support: KeyError: 'PNG' in Image.save()
# Do not move this block below other imports.
# ---------------------------------------------------------------------------
import importlib
import sys
from pathlib import Path


def _pin(*module_names):
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            sys.modules.setdefault(name, mod)
        except ImportError:
            pass  # module not installed; nothing to pin


_pin(
    "numpy",
    "numpy._core",
    "numpy._core._multiarray_umath",
    "PIL",
    "PIL.Image",
    "PIL.PngImagePlugin",  # registers SAVE["PNG"] at import time
    "matplotlib",
    "matplotlib.gridspec",
    "matplotlib.backends.backend_agg",
    "pyarrow",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
