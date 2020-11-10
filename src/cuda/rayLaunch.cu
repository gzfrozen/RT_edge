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

/* change screen space position into ray direction */
__forceinline__ __host__ __device__ vec3f screen_to_direction(const vec2f &screen,
                                                              const vec3f &direction,
                                                              const vec3f &horizontal,
                                                              const vec3f &vertical)
{
    return normalize(direction + (screen.x - 0.5f) * horizontal + (screen.y - 0.5f) * vertical);
}

// calculate edge strength, used in classic renderer
__forceinline__ __host__ __device__ float get_edge_strength(const int &M, const int &i)
{
    return 1.f - 2.f * fabsf((float)i - (float)M / 2.f) / (float)M;
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__fastRenderer()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;
    const int &accumID = optixLaunchParams.frame.accumID;
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

extern "C" __global__ void __raygen__classicRenderer()
{
    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;
    const auto &camera = optixLaunchParams.camera;
    const int &N = optixLaunchParams.classic.RAY_STENCIL_QUALITY.x;
    const int &n = optixLaunchParams.classic.RAY_STENCIL_QUALITY.y;
    const float &radius = optixLaunchParams.classic.RAY_STENCIL_RADIUS;
    const int &stencil_length = optixLaunchParams.classic.stencil_length;
    const vec2f *const ray_stencil = optixLaunchParams.classic.ray_stencil;

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    // normalized screen plane position, in [0,1]^2
    vec2f screen = (vec2f(ix + 0.5f, iy + 0.5f) / vec2f(optixLaunchParams.frame.size));
    // main ray direction
    vec3f main_rayDir = screen_to_direction(screen, camera.direction, camera.horizontal, camera.vertical);

    // values for caculating edge strength
    bool main_is_hit{false};
    uint32_t main_geometryID;
    int num_missed;
    int num_different;

    // tracing center ray
    {
        PRD_Classic prd_main;
        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer(&prd_main, u0, u1);
        optixTrace(optixLaunchParams.traversable,
                   camera.position,
                   main_rayDir,
                   0.f,   // tmin
                   1e20f, // tmax
                   0.0f,  // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   CLASSIC_RAY_TYPE,              // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   CLASSIC_RAY_TYPE,              // missSBTIndex
                   u0, u1);
        if (prd_main.is_hit)
        {
            main_is_hit = true;
            main_geometryID = prd_main.geometryID;
        }
    }

    // tracing ray_stencil
    for (int i = 0; i < stencil_length; i++)
    {
        vec3f sub_rayDir = screen_to_direction(screen + ray_stencil[i], camera.direction, camera.horizontal, camera.vertical);
        PRD_Classic prd_sub;
        // the values we store the PRD pointer in:
        uint32_t u0, u1;
        packPointer(&prd_sub, u0, u1);
        optixTrace(optixLaunchParams.traversable,
                   camera.position,
                   sub_rayDir,
                   0.f,   // tmin
                   1e20f, // tmax
                   0.0f,  // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
                   CLASSIC_RAY_TYPE,              // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   CLASSIC_RAY_TYPE,              // missSBTIndex
                   u0, u1);
        if (prd_sub.is_hit)
        {
            if (prd_sub.geometryID != main_geometryID)
            {
                num_different++;
            }
        }
        else
        {
            num_missed++;
        }
    }

    // calculate edge
    vec3f pixelColor = {255.f};
    float edge_strength{0.f};
    edge_strength = main_is_hit ? get_edge_strength(stencil_length, num_different + num_missed)
                                : get_edge_strength(stencil_length, num_missed);
    pixelColor *= 1 - edge_strength;

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