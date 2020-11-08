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

/* change polor into xy coordinate vector */
__forceinline__ __host__ __device__ vec2f polor_to_normal(const float &theta, const float &r)
{
    return vec2f(cosf(theta) * r, sinf(theta) * r);
}

/* make ray_stencil[] into an array stored directions of a ray stencil.
*ray_stencil: pointer of a vec2f ray directions array (screen space),
center_pixel_location: location of center pixel (screen space),
h: radius (screen space),
N: number of circles,
n: number of rays in first circle, must be multiple of 4. number doubles each time on next circle */
__host__ __device__ void makeRayStencil1(vec2f *const ray_stencil,
                                         const vec2f &center_pixel_location,
                                         const float &h,
                                         const int &N,
                                         const int &n)
{
    assert(h > 0.f);
    assert(N > 0);
    assert(n > 0 && n % 4 == 0);

    float temp_r;
    int temp_n{n};
    float theta;
    int index{0};
    for (int i = 0; i < N; i++)
    {
        temp_r = h / N * (i + 1);
        for (int j = 0; j < temp_n; j++)
        {
            theta = 2 * M_PI * j / temp_n;
            ray_stencil[index + j] = polor_to_normal(temp_r, theta) + center_pixel_location;
        }
        index += temp_n;
        temp_n *= 2;
    }
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
    const auto &camera = optixLaunchParams.camera;
    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;
    const vec2i &quality = optixLaunchParams.parameters.RAY_STENCIL_QUALITY;
    const float &radius = optixLaunchParams.parameters.RAY_STENCIL_RADIUS;

    // normalized screen plane position, in [0,1]^2
    vec2f screen;
    screen = (vec2f(ix + 0.5f, iy + 0.5f) / vec2f(optixLaunchParams.frame.size));

    vec3f rayDir = screen_to_direction(screen, camera.direction, camera.horizontal, camera.vertical);

    // // just try to calculate (2^N - 1) * n, cuda seems doesn't suppor pow(int, int)
    // int array_size{1};
    // for (int i = 0; i < quality.x; i++)
    // {
    //     array_size *= 2;
    // }
    // array_size -= 1;
    // array_size *= quality.y;

    // vec2f *rayStencil;
    // rayStencil = new vec2f[array_size];
    // makeRayStencil1(rayStencil, screen, radius, quality.x, quality.y);

    // // generate ray direction
    // vec3f *rayDir;
    // rayDir = new vec3f[array_size + 1];
    // for (int i = 0; i < array_size; i++)
    // {
    //     rayDir[i] = screen_to_direction(rayStencil[i], camera.direction, camera.horizontal, camera.vertical);
    // }
    // rayDir[array_size] = screen_to_direction(screen, camera.direction, camera.horizontal, camera.vertical);

    // // to do: ray tracing..

    // delete[] rayStencil;
    // delete[] rayDir;
    return vec3f(255.f, 255.f, 255.f);
}
