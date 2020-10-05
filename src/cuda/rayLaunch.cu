#include "CUDA.hpp"

/* change spherical vector into normalized xyz vector*/
__forceinline__ __host__ __device__ vec3f sphere_to_normal(const vec3f &sphere_direction)
{
    const float &theta = sphere_direction.y;
    const float &phi = sphere_direction.z;
    return vec3f(cos(theta) * sin(phi),
                 sin(theta) * sin(phi),
                 cos(phi));
}

/* change xyz vector into spherical vector with 0 length */
// __forceinline__ __host__ __device__ vec3f normal_to_sphere(const vec3f &xyz_direction)
// {
//     float theta = atan(xyz_direction.y / xyz_direction.x);
//     float phi = acos(xyz_direction.z / length(xyz_direction));
//     if (xyz_direction.x < 0)
//     {
//         theta = (xyz_direction.y > 0) ? theta + M_PI : theta - M_PI;
//     }
//     return vec3f(0.f, theta, phi);
// }

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int accumID = optixLaunchParams.frame.accumID;
    const auto &camera = optixLaunchParams.camera;

    PRD prd;
    prd.random.init(ix + accumID * optixLaunchParams.frame.size.x,
                    iy + accumID * optixLaunchParams.frame.size.y);
    prd.pixelColor = vec3f(255.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);

    int numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;

    vec3f pixelColor = 0.f;
    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
    {
        vec3f rayDir;
        if (camera.camera_type == PINHOLE)
        {
            // normalized screen plane position, in [0,1]^2
            vec2f screen;
            if (optixLaunchParams.parameters.NUM_PIXEL_SAMPLES > 1)
            {
                screen = (vec2f(ix + prd.random() - 0.5f, iy + prd.random() - 0.5f) / vec2f(optixLaunchParams.frame.size));
            }
            else
            {
                screen = (vec2f(ix, iy) / vec2f(optixLaunchParams.frame.size));
            }

            // generate ray direction
            rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
        }
        else if (camera.camera_type == ENV)
        {
            // sperical coordinate position
            vec3f spherical_position;
            if (optixLaunchParams.parameters.NUM_PIXEL_SAMPLES > 1)
            {
                spherical_position = ((ix + prd.random() - 0.5f) * camera.horizontal + (iy + prd.random() - 0.5f) * camera.vertical);
            }
            else
            {
                spherical_position = ((float)ix * camera.horizontal + (float)iy * camera.vertical);
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

    const int r = int(255.99f * min(pixelColor.x / numPixelSamples, 1.f));
    const int g = int(255.99f * min(pixelColor.y / numPixelSamples, 1.f));
    const int b = int(255.99f * min(pixelColor.z / numPixelSamples, 1.f));

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}