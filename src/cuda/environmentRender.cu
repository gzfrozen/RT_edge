#include "CUDA.hpp"

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow()
{
    /* not going to be used ... */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
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

//------------------------------------------------------------------------------
// dummy ray gen program - the real ray gen program is in rayLauch.cu
//------------------------------------------------------------------------------

// extern "C" __global__ void __raygen__dummy()
// {
// }