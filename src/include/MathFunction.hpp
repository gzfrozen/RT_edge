#pragma once

#include "gdt/math/vec.h"

/* change spherical vector into normalized xyz vector*/
__host__ __device__ inline vec3f sphere_to_normal(const vec3f &sphere_direction)
{
    const float &theta = sphere_direction.y;
    const float &phi = sphere_direction.z;
    return vec3f(cos(theta) * sin(phi),
                 sin(theta) * sin(phi),
                 cos(phi));
}

/* change xyz vector into spherical vector with 0 length */
__host__ __device__ inline vec3f normal_to_sphere(const vec3f &xyz_direction)
{
    float theta = atan(xyz_direction.y / xyz_direction.x);
    float phi = acos(xyz_direction.z / length(xyz_direction));
    if (xyz_direction.x < 0)
    {
        theta = (xyz_direction.y > 0) ? theta + M_PI : theta - M_PI;
    }
    return vec3f(0.f, theta, phi);
}