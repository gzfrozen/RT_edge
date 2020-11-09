#include "CUDA.hpp"

extern "C" __device__ vec3f __continuation_callable__fast_launch(const int &ix, const int &iy)
{
    return vec3f{0.f, 0.f, 0.f}; // not using
}

extern "C" __device__ vec3f __continuation_callable__classic_launch(const int &ix, const int &iy)
{
    return vec3f{0.f, 0.f, 0.f}; // not using
}