#include "CUDA.hpp"

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__fastRenderer()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;

    vec3f pixelColor = 0.f;

    if (optixLaunchParams.parameters.RENDERER_TYPE == FAST)
    {
        pixelColor = optixContinuationCall<vec3f, int const &, int const &>(FAST, ix, iy);
    }
    else if (optixLaunchParams.parameters.RENDERER_TYPE == CLASSIC)
    {
        pixelColor = optixContinuationCall<vec3f, int const &, int const &>(CLASSIC, ix, iy);
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
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int &numPixelSamples = optixLaunchParams.parameters.NUM_PIXEL_SAMPLES;

    vec3f pixelColor = {0.f, 255.f, 0.f};
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