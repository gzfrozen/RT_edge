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

#pragma once

#include "gdt/math/vec.h"
#include "gdt/math/LinearSpace.h"
#include <optix7.hpp>
#include <ControlParams.hpp>

using namespace gdt;

struct TriangleMeshSBTData
{
    uint32_t geometryID;
    vec3f color;
    vec3f *vertex;
    vec3f *normal;
    vec2f *texcoord;
    vec3i *index;
    bool hasTexture;
    cudaTextureObject_t texture;
};

struct LaunchParams
{
    struct
    {
        uint32_t *colorBuffer;
        vec2i size;
        int accumID{0};
    } frame;

    struct
    {
        int camera_type;
        vec3f position;
        vec3f direction;
        vec3f horizontal;
        vec3f vertical;
        linear3f matrix;
    } camera;

    struct
    {
        vec3f origin, du, dv, power;
    } light;

    struct
    {
        /* radius (screen space) */
        float RAY_STENCIL_RADIUS{1e-2f};

        /* number of circles, number of rays in first circle */
        /* number of rays must be multiple of 4. */
        /* number doubles each time on next circle */
        vec2i RAY_STENCIL_QUALITY{2, 8};

        /* threshold of crease edges */
        float NORMAL_CHANGE_THRESHOLD{M_PI * 2.f / 3.f};
        /* threshold of self-occluding silhouettes */
        float DISTANCE_CHANGE_THRESHOLD{0.2f};

        /* screen space offside of each ray in the ray stencil */
        vec2f ray_stencil[256];
        /* number of rays in the stencil */
        int stencil_length{0};
        /* the indices of 4 rays that normal is concerned */
        int stencil_normal_index[4];
    } classic;

    OptixTraversableHandle traversable;

    // parameters
    struct
    {
        int RENDERER_TYPE{FAST};
        int LAUNCH_RAY_TYPE{RADIANCE_RAY_TYPE};

        int NUM_LIGHT_SAMPLES{16};
        int NUM_PIXEL_SAMPLES{1};

        float WAVE_LENGTH{100.f};

        float EDGE_DETECTION_DEPTH{5e-2f};
        float MAX_EDGE_DISTANCE{2e-1f};
        float MIN_EDGE_ANGLE{0.f};
        float MAX_EDGE_ANGLE{M_PI / 2.f};
        bool OVER_PI_EDGE{false};
    } parameters;
};