// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "CUDA.hpp"

typedef struct
{
    float r; // a fraction between 0 and 1
    float g; // a fraction between 0 and 1
    float b; // a fraction between 0 and 1
} RGB;

typedef struct
{
    float h; // angle in degrees
    float s; // a fraction between 0 and 1
    float v; // a fraction between 0 and 1
} HSV;

__forceinline__ __host__ __device__ RGB hsv2rgb(const HSV &in)
{
    float hh, p, q, t, ff;
    int i;
    RGB out;

    if (in.s <= 0.0)
    { // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if (hh >= 360.0)
        hh = 0.0;
    hh /= 60.0;
    i = (int)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch (i)
    {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;
}

//------------------------------------------------------------------------------
// closest hit programs.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but in some simple situations,
// only dummy programms is needed (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{
    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f &A = sbtData.vertex[index.x];
    const vec3f &B = sbtData.vertex[index.y];
    const vec3f &C = sbtData.vertex[index.z];
    vec3f Ng = cross(B - A, C - A);
    vec3f Ns = (sbtData.normal)
                   ? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                   : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f)
        Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord)
    {
        const vec2f tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= (vec3f)fromTexture;
    }

    // start with some ambient term
    vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const vec3f surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];
    const int numLightSamples = NUM_LIGHT_SAMPLES;
    for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++)
    {
        // produce random light sample
        const vec3f lightPos = optixLaunchParams.light.origin + prd.random() * optixLaunchParams.light.du + prd.random() * optixLaunchParams.light.dv;
        vec3f lightDir = lightPos - surfPos;
        float lightDist = gdt::length(lightDir);
        lightDir = normalize(lightDir);

        // trace shadow ray:
        const float NdotL = dot(lightDir, Ns);
        if (NdotL >= 0.f)
        {
            vec3f lightVisibility = 0.f;
            // the values we store the PRD pointer in:
            uint32_t u0, u1;
            packPointer(&lightVisibility, u0, u1);
            optixTrace(optixLaunchParams.traversable,
                       surfPos + 1e-3f * Ng,
                       lightDir,
                       1e-3f,                     // tmin
                       lightDist * (1.f - 1e-3f), // tmax
                       0.0f,                      // rayTime
                       OptixVisibilityMask(255),
                       // For shadow rays: skip any/closest hit shaders and terminate on first
                       // intersection with anything. The miss shader is used to mark if the
                       // light was visible.
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                       SHADOW_RAY_TYPE, // SBT offset
                       RAY_TYPE_COUNT,  // SBT stride
                       SHADOW_RAY_TYPE, // missSBTIndex
                       u0, u1);
            pixelColor += lightVisibility * optixLaunchParams.light.power * diffuseColor * (NdotL / (lightDist * lightDist * numLightSamples));
        }
    }

    prd.pixelColor = pixelColor;
}

extern "C" __global__ void __closesthit__shadow()
{
    /* not going to be used ... */
}

extern "C" __global__ void __closesthit__phase()
{
    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // phase and color calculation
    // ------------------------------------------------------------------
    const float distance = optixGetRayTmax();
    const float phase = fmod(distance, WAVE_LENGTH) * 360.f / WAVE_LENGTH;
    const HSV hsv = {phase, 1.f, 0.7f}; // use hsv color space
    const RGB rgb = hsv2rgb(hsv);

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f &A = sbtData.vertex[index.x];
    const vec3f &B = sbtData.vertex[index.y];
    const vec3f &C = sbtData.vertex[index.z];
    vec3f Ng = cross(B - A, C - A);
    vec3f Ns = (sbtData.normal)
                   ? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                   : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f)
        Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // final result mixed with some simple ambient effect
    // ------------------------------------------------------------------
    prd.pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) + vec3f(rgb.r, rgb.g, rgb.b);
}