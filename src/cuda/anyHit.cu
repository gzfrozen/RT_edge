#include "CUDA.hpp"

//------------------------------------------------------------------------------
// any hit programs.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but in some simple situations,
// only dummy programms is needed (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------
extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow()
{
    /* not going to be used ... */
}

extern "C" __global__ void __anyhit__phase()
{
    /* not going to be used ... */
}