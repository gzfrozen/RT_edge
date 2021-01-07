# RT_edge

### About this project

This project changed its name from 'EM_tracing' to 'RT-edge'.
This project means to develop a new method of triangle-based 3D geometry edge detection algorithm.
No related implementation at this point.

## Introduction

RT_edge is a software based on [Nvidia Optix Ray Tracing Engine](https://developer.nvidia.com/optix). A big part of the code came from [ingowald/optix7course](https://github.com/ingowald/optix7course).

## Building the code

This code requires [CMake](https://cmake.org/download/) (version 3.18 or higher) to build. MSVC 2019 and Clang 10.0.0 C++ compiler under Windows 10 has been tested to build successfully. (Theoretically other combinations of compilers and OS like GCC under Linux should also work)

## Dependencies

- A Nvidia GPU
- a compiler
  - on Windows 10, tested with MSVC 2019 and Clang 10.0.0 C++ compiler
  - other common compilers (not tested)
- [CMake](https://cmake.org/download/) (version 3.18 or higher)
- [CUDA 10.x or CUDA 11.x](https://developer.nvidia.com/cuda-downloads)
- Latest Nvidia GPU Driver
- [OptiX 7.2 SDK](https://developer.nvidia.com/designworks/optix/download)
- a 3D geometry object file
  - recommend Rungholt, Crytek Sponza, BMW on [https://casual-effects.com/data/](https://casual-effects.com/data/)

## Important settings and parameters

- In /CMakeLists.txt, CUDA_ARCHITECTURES. Change it to match your GPU.

  - see details in [CMake Document](https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES), [CUDA Document](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)

  >     set_target_properties(CudaPTX PROPERTIES
  >     ...
  >     CUDA_ARCHITECTURES 75-real # Set your GPU architecture (important)
  >     ...
  >     )

- Also in /CMakeLists.txt, \_OBJ_FILE. Change it to match your .obj file path
  >     #Set your OBJ file path
  >     target_compile_definitions(RT_edge PUBLIC _OBJ_FILE="${PROJECT_SOURCE_DIR}/models/sponza.obj")

## Controlling

- Use left, middle, right mouse button to drag and scroll
- Press w, s, a, d to move
- (New) Press ESC to toggle GUI parameter control pannel
- Press =, - to control drag speed, alt + =, - to control move speed
- Press p, l to switch between pin-hole camera and 360 degree camera (not working properly at the moment)
- Press r, switch to normal rendering mode
- Press t, switch to phase detection mode
- (New) Press e, switch to fast feature line mode
- (New) Press q, switch to classic feature line mode
- (New) Press m, switch to mixed (normal + fast feature line) rendering mode
- Press c to check current camera information
