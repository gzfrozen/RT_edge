#include "CUDA.hpp"

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.pixelColor = vec3f(1.f);
}

extern "C" __global__ void __miss__shadow()
{
    // we didn't hit anything, so the light is visible
    vec3f &prd = *getPRD<vec3f>();
    prd = vec3f(1.f);
}

extern "C" __global__ void __miss__phase()
{
    // set to constant white as background color
    PRD &prd = *getPRD<PRD>();
    prd.pixelColor = vec3f(1.f);
}

extern "C" __global__ void __miss__mono()
{
    PRD &prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.pixelColor = vec3f(1.f);
}

extern "C" __global__ void __miss__edge()
{
    // we didn't hit anything, so it is not the edge
    PRD_Edge &prd_edge = *getPRD<PRD_Edge>();
    prd_edge.is_edge = false;
}