# ---------------------------------------------------------------------------
# Pin one-init C extensions and lazy-registered plugins in sys.modules before
# pytest-cov's SysModuleSaver snapshots sys.modules. This conftest.py is
# loaded during pytest's initial conftest phase, before pytest-cov takes its
# sys.modules snapshot, so pinned modules survive eviction-and-reimport:
#   - numpy 2.x: "cannot load module more than once per process"
#   - Pillow PNG support: KeyError: 'PNG' in Image.save()
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
            pass


_pin(
    "numpy",
    "numpy._core",
    "numpy._core._multiarray_umath",
    "PIL",
    "PIL.Image",
    "PIL.PngImagePlugin",
    "matplotlib",
    "matplotlib.gridspec",
    "matplotlib.backends.backend_agg",
    "pyarrow",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
