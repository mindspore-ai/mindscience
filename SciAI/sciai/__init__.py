"""sciai modules"""
from sciai import operators, architecture, common, context, utils
from sciai.version import __version__
from sciai.model import AutoModel

__all__ = []
__all__.extend(__version__)
__all__.extend(architecture.__all__)
__all__.extend(common.__all__)
__all__.extend(context.__all__)
__all__.extend(operators.__all__)
__all__.extend(utils.__all__)
__all__.extend(["AutoModel"])
