"""
conftest.py — project-root pytest configuration.

Blocks the `langsmith` pytest plugin which conflicts with pydantic v1 on Python 3.12.
This is a system-level incompatibility in the user's pre-installed packages; it does
not affect our code.
"""
# Prevent langsmith from auto-loading its broken pytest plugin on Python 3.12
import sys
import os

# Unregister langsmith entrypoint before it crashes pytest startup
if sys.version_info >= (3, 12):
    try:
        import importlib.metadata as _meta
        # Monkey-patch: override the langsmith entry_points so pytest doesn't load it
        _orig = _meta.entry_points

        def _patched_entry_points(**kwargs):
            eps = _orig(**kwargs)
            if hasattr(eps, '__iter__'):
                return [ep for ep in eps
                        if not (hasattr(ep, 'group') and ep.group == 'pytest11'
                                and 'langsmith' in ep.value)]
            return eps

        _meta.entry_points = _patched_entry_points
    except Exception:
        pass

