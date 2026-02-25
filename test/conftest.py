import sys
import os

# Expose the build directory so the compiled C++ extension is importable.
_builddir = os.path.join(os.path.dirname(__file__), "..", "builddir")
if _builddir not in sys.path:
    sys.path.insert(0, _builddir)
