#pragma once

#include <SampleRenderer.hpp>
#include <GLFWindow.h>

class MainWindow : public GLFCameraWindow
{
public:
    MainWindow(const std::string &title,
               const Model *model,
               const int &launch_ray_type,
               const Camera &camera,
               const QuadLight &light,
               const float worldScale);
    ~MainWindow();

private:
    inline Camera get_camera() const
    {
        return Camera{cameraFrame.get_from(),
                      cameraFrame.get_at(),
                      cameraFrame.get_up(),
                      cameraFrame.get_frame()};
    }
    void before_render() override;
    void render() override;
    void draw() override;
    void resize(const vec2i &newSize) override;

    /*! draw GUI layout */
    void draw_gui();

    vec2i fbSize;
    GLuint fbTexture{0};
    SampleRenderer sample;
    std::vector<uint32_t> pixels;
};