project(
  'NilVec',
  'cpp',
  version: '0.1.0',
  meson_version: '>= 1.3.0',
  default_options: [
    'warning_level=3',
    'cpp_std=c++23',
    'buildtype=release',
    'b_lto=true',
    'track_conflicts=true',
    'use_hdf5=true',
  ],
)

# Build options
track_conflicts = get_option('track_conflicts')
use_hdf5 = get_option('use_hdf5')

# Thread support
thread_dep = dependency('threads')

dependencies = [thread_dep]

# HDF5 support (optional, for loading standard datasets)
if use_hdf5
  hdf5_dep = dependency('hdf5', language: 'cpp', required: false)
  if hdf5_dep.found()
    dependencies += hdf5_dep
  else
    warning('HDF5 not found. Dataset loading will be limited.')
    use_hdf5 = false
  endif
endif

# Compile arguments
cpp_args = []
if track_conflicts
  cpp_args += '-DNILVEC_TRACK_CONFLICTS'
endif
if use_hdf5
  cpp_args += '-DNILVEC_USE_HDF5'
endif

# Enable architecture-specific optimizations (AVX2, NEON, etc.)
cpp_args += '-march=native'

exe = executable(
  'NilVec',
  'main.cpp',
  install: true,
  dependencies: dependencies,
  cpp_args: cpp_args,
)

test('basic', exe)

# Python bindings
py_mod = import('python')
py = py_mod.find_installation(required: false, pure: false)

if py.found()
  pybind11_dep = dependency('pybind11', required: true)
  
  if pybind11_dep.found()
    py.extension_module(
      '_nilvec',
      'src/python_bindings.cpp',
      dependencies: dependencies + [pybind11_dep],
      cpp_args: cpp_args,
      subdir: 'nilvec',
      install: true,
    )
    
    py.install_sources(
      ['nilvec/__init__.py', 'nilvec/benchmark.py'],
      subdir: 'nilvec',
    )
  else
    warning('pybind11 not found, skipping Python bindings')
  endif
endif
