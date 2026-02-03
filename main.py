#!/usr/bin/env python3
import sys
import os

# Ensure we can find the local package
sys.path.append(os.getcwd())

# Add build directory to path for the extension
build_dir = os.path.join(os.getcwd(), "builddir")
sys.path.append(build_dir)

try:
    from nilvec.benchmark import run_benchmark
except ImportError as e:
    print(f"Error importing nilvec.benchmark: {e}")
    # try direct import for dev
    try:
        from nilvec.benchmark import run_benchmark
    except ImportError:
        print("Please run `meson compile -C builddir` first.")
        sys.exit(1)

if __name__ == "__main__":
    run_benchmark()
