#pragma once

enum camera_type
{
    PINHOLE,
    ENV
};

// we have three ray types
enum
{
    RADIANCE_RAY_TYPE = 0,
    SHADOW_RAY_TYPE,
    PHASE_RAY_TYPE,
    RAY_TYPE_COUNT
};