#pragma once

enum renderer_type
{
    FAST = 0,
    CLASSIC,
    MIXED,
    RENDERER_TYPE_COUNT
};

enum camera_type
{
    PINHOLE = 0,
    ENV
};

// we have five ray types
enum
{
    RADIANCE_RAY_TYPE = 0,
    SHADOW_RAY_TYPE,
    PHASE_RAY_TYPE,
    MONO_RAY_TYPE,
    EDGE_RAY_TYPE,
    CLASSIC_RAY_TYPE,
    RAY_TYPE_COUNT
};