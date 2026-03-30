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
  ],
)

# Build options
track_conflicts = get_option('track_conflicts')
python_bindings = get_option('python_bindings')

# Thread support
thread_dep = dependency('threads')

hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

dependencies = [thread_dep, hdf5_dep]

# Compile arguments
cpp_args = ['-DNILVEC_USE_HDF5']
if track_conflicts
  cpp_args += '-DNILVEC_TRACK_CONFLICTS'
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

if python_bindings and py.found()
  pybind11_dep = dependency('pybind11', required: true)
  
  if pybind11_dep.found()
    py.extension_module(
      '_nilvec',
      'src/python_bindings.cpp',
      dependencies: dependencies + [pybind11_dep],
      cpp_args: cpp_args,
      install: true,
    )
    
    py.install_sources(
      ['nilvec.py'],
    )
  else
    warning('pybind11 not found, skipping Python bindings')
  endif
endif
