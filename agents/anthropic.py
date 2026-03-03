from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_root_shim = Path(__file__).resolve().parent.parent / "anthropic.py"
_spec = spec_from_file_location("anthropic_root_shim", _root_shim)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

Anthropic = _module.Anthropic

__all__ = ["Anthropic"]
