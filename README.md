# EM_tracing

### About the name

'EM' means 'Electromagnetic Waves'. However not many EM related functions at this point.

## Introduction

EM_tracing is a software based on [Nvidia Optix Ray Tracing Engine](https://developer.nvidia.com/optix). A big part of the code came from [ingowald/optix7course](https://github.com/ingowald/optix7course).

## Building the code

This code requires [CMake](https://cmake.org/download/) (version 3.18 or higher) to build. MSVC 2019 and Clang 10.0.0 C++ compiler under Windows 10 has been tested to build successfully. (Theoretically other combinations of compilers and OS like GCC under Linux should also work)

## Dependencies

 - A Nvidia GPU
 - a compiler
	 - on Windows 10, tested with MSVC 2019 and Clang 10.0.0 C++ compiler
	 - other common compilers (not tested)
 - [CMake](https://cmake.org/download/) (version 3.18 or higher)
 - [CUDA 10.x or CUDA 11](https://developer.nvidia.com/cuda-downloads)
 - Latest Nvidia GPU Driver
 - [OptiX 7.1 SDK](https://developer.nvidia.com/designworks/optix/download) 
 - a 3D geometry object file
	 - recommend Crytek Sponza, BMW on [https://casual-effects.com/data/](https://casual-effects.com/data/)

## Important settings and parameters 

- In /CMakeLists.txt, CUDA_ARCHITECTURES. Change it to match your GPU.
	- see details in [CMake Document](https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES), [CUDA Document](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)

>     set_target_properties(CudaPTX PROPERTIES
>     ...
>     CUDA_ARCHITECTURES 75-real # Set your GPU architecture (important)
>     ...
>     )
- Also in /CMakeLists.txt, _OBJ_FILE. Change it to match your .obj file path
>     #Set your OBJ file path
>     target_compile_definitions(EM_tracing PUBLIC _OBJ_FILE="${PROJECT_SOURCE_DIR}/models/sponza.obj")

## Controlling 

- Use left, middle, right mouse button to drag and scroll
- Press w, s, a, d to move
- Press =, - to control drag speed, alt + =, - to control move speed
- Press p, l to switch between pin-hole camera and 360 degree camera (not working properly at the moment)
- Press r, t to switch between normal rendering mode and phase detection mode
- Press c to check current camera information
