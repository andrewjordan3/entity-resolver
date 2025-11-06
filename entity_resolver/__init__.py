# Expose the main classes to the top level of the package
from .config import ResolverConfig
from .resolver import EntityResolver

# Define the package version
__version__ = '0.1.0'

__all__ = ['ResolverConfig', 'EntityResolver']
