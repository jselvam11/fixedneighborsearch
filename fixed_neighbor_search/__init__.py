# First import the C++ extension functions
try:
    # Import compiled C++ functions directly
    from _fixedneighborsearch import fixed_radius_search, build_spatial_hash_table
    __all__ = ['FixedRadiusSearch', 'fixed_radius_search', 'build_spatial_hash_table']
except ImportError:
    # Handle the case when extension is not built yet
    __all__ = ['FixedRadiusSearch']

# Then import the Python wrapper
from .neighbor_search import FixedRadiusSearch