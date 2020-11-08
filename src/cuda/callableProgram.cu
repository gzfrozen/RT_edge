#include "CUDA.hpp"

/* change spherical vector into normalized xyz vector*/
__forceinline__ __host__ __device__ vec3f sphere_to_normal(const vec3f &sphere_direction)
{
    const float &theta = sphere_direction.y;
    const float &phi = sphere_direction.z;
    return vec3f(cosf(theta) * sinf(phi),
                 sinf(theta) * sinf(phi),
                 cosf(phi));
}

extern "C" __device__ vec3f __continuation_callable__fast_launch(const int &ix, const int &iy)
{
    const int &accumID = optixLaunchParams.frame.accumID;
    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;
    const auto &camera = optixLaunchParams.camera;

    PRD prd;
    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
                    iy + accumID * optixLaunchParams.frame.size.y);
    prd.pixelColor = vec3f(255.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    vec3f pixelColor = 0.f;
    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
    {
        vec3f rayDir;
        if (camera.camera_type == PINHOLE)
        {
            // normalized screen plane position, in [0,1]^2
            vec2f screen;
            if (numPixelSamples > 1)
            {
                screen = (vec2f(ix + prd.random(), iy + prd.random()) / vec2f(optixLaunchParams.frame.size));
            }
            else
            {
                screen = (vec2f(ix + 0.5f, iy + 0.5f) / vec2f(optixLaunchParams.frame.size));
            }

            // generate ray direction
            rayDir = screen_to_direction(screen, camera.direction, camera.horizontal, camera.vertical);
        }
        else if (camera.camera_type == ENV)
        {
            // sperical coordinate position
            vec3f spherical_position;
            if (numPixelSamples > 1)
            {
                spherical_position = ((ix + prd.random()) * camera.horizontal + (iy + prd.random()) * camera.vertical);
            }
            else
            {
                spherical_position = ((ix + 0.5f) * camera.horizontal + (iy + 0.5f) * camera.vertical);
            }
            spherical_position -= vec3f(0.f, M_PI, 0.f);
            // change into xyz coordinate position
            const vec3f xyz_position(sphere_to_normal(spherical_position));
            // view port transform
            rayDir = {dot(camera.matrix.vx, xyz_position),
                      dot(camera.matrix.vy, xyz_position),
                      dot(camera.matrix.vz, xyz_position)};
        }

        const int &ray_type = optixLaunchParams.parameters.LAUNCH_RAY_TYPE;
        optixTrace(optixLaunchParams.traversable,
                   camera.position,
                   rayDir,
                   0.f,   // tmin
                   1e20f, // tmax
                   0.0f,  // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   ray_type,                      // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   ray_type,                      // missSBTIndex
                   u0, u1);
        pixelColor += prd.pixelColor;
    }
    return pixelColor;
}

extern "C" __device__ vec3f __continuation_callable__classic_launch(const int &ix, const int &iy)
{
    return vec3f(255.f, 255.f, 255.f);
}
