# entity_resolver/utils/clean_mem.py
"""
GPU Memory Management Utilities for RAPIDS and CuPy.

This module provides robust, reusable tools for managing GPU memory within
the RAPIDS ecosystem. The primary motivation is to prevent memory leaks and
crashes in long-running processes or complex applications by providing a
guaranteed cleanup mechanism.

The core utility is the `@gpu_memory_cleanup` decorator, which can be applied
to any function or method that creates GPU objects (e.g., cuDF DataFrames,
CuPy arrays). It ensures that after the function finishes, all unused, cached
memory is released from the underlying allocators.

This is critical because Python's garbage collector alone is often not
sufficient to free GPU memory, as the memory pools managed by RMM and CuPy
can retain cached blocks for performance reasons. This decorator forces those
pools to release the memory, making it available for subsequent operations.

Public API:
    - gpu_memory_cleanup: A decorator for functions and methods.
    - manual_gpu_cleanup: A function to trigger cleanup manually, useful in
                          interactive sessions like Jupyter notebooks.
"""

import gc
import logging
import cupy
import rmm
from functools import wraps
from typing import Callable, Any

# Set up a logger for decorator-specific messages.
logger = logging.getLogger(__name__)

# --- Public API ---

def gpu_memory_cleanup(func: Callable) -> Callable:
    """
    A robust decorator that ensures GPU memory cleanup after function execution.

    This decorator is designed for functions that use CuPy and the RAPIDS
    ecosystem. It works seamlessly with both standalone functions and class methods.

    The decorator guarantees the following cleanup actions in a `finally` block,
    ensuring they run even if the decorated function raises an exception:
      1. Synchronizes the default CUDA stream to ensure all kernels are complete.
      2. Runs Python's garbage collector to release object references.
      3. Releases all unused, cached memory from the RMM and CuPy allocators.

    Example Usage:
        @gpu_memory_cleanup
        def my_gpu_intensive_task():
            # ... create cudf DataFrames and cupy arrays ...
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            # Call the core cleanup logic with default settings.
            _cleanup_gpu_memory(synchronize=True, run_gc=True, release_pools=True)
    return wrapper

def manual_gpu_cleanup():
    """
    Manually triggers the full GPU memory cleanup process.

    This is useful for freeing GPU memory in interactive environments like
    Jupyter notebooks or an IPython console after performing large computations.
    """
    logger.debug("Manual GPU memory cleanup initiated.")
    _cleanup_gpu_memory(synchronize=True, run_gc=True, release_pools=True)

# --- Internal Implementation ---

def _cleanup_gpu_memory(synchronize: bool, run_gc: bool, release_pools: bool):
    """
    Core cleanup logic separated for reusability and precise control.

    Each step is wrapped in a try/except block to ensure that a failure in one
    stage does not prevent subsequent stages from running.
    """
    # Step 1: Synchronize CUDA operations to wait for pending kernels.
    if synchronize:
        try:
            cupy.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.warning(f"Error during CUDA stream sync (non-critical): {e}")

    # Step 2: Run garbage collection to free Python-level object references.
    if run_gc:
        gc.collect()

    # Step 3: Release memory from the underlying memory pools.
    if release_pools:
        # Check if CuPy is configured to use the RMM allocator.
        is_rmm_allocator_for_cupy = (cupy.cuda.get_allocator() is rmm.rmm_cupy_allocator)

        # Always attempt to release the RMM pool, as it may be used by
        # libraries like cuDF even if CuPy is using its own pool.
        try:
            initial_mr = rmm.mr.get_current_device_resource()
            releasable_mr = _find_releasable_rmm_resource(initial_mr)
            if releasable_mr is not None:
                releasable_mr.release()
                logger.debug("RMM memory pool released.")
        except Exception as e:
            logger.warning(f"Error during RMM pool release (non-critical): {e}")

        # If CuPy is using its own separate allocator, we must clean it up too.
        if not is_rmm_allocator_for_cupy:
            logger.debug(
                "CuPy is not using RMM allocator. Freeing CuPy pools independently."
            )
            try:
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                logger.warning(f"Error freeing CuPy default pool (non-critical): {e}")
            try:
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                logger.warning(f"Error freeing CuPy pinned pool (non-critical): {e}")

def _find_releasable_rmm_resource(mr):
    """
    Traverse RMM's wrapped resources to find the actual pool allocator.
    
    RMM can wrap resources (e.g., for logging or tracking stats), so this function
    traverses the chain of upstream resources to find the underlying object that
    exposes a `.release()` method.
    """
    seen_resources = set()
    current_resource = mr
    while current_resource is not None and id(current_resource) not in seen_resources:
        seen_resources.add(id(current_resource))
        
        if hasattr(current_resource, "release"):
            return current_resource
        
        # Introspect to find the upstream resource, covering different RMM versions.
        upstream_attr = getattr(current_resource, "upstream_mr", None) or \
                        getattr(current_resource, "get_upstream", None)
        
        if callable(upstream_attr):
            current_resource = upstream_attr()
        else:
            current_resource = upstream_attr
            
    return None
