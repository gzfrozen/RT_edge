#pragma once

#include <Renderer.hpp>
#include <GLFWindow.h>
#include <JSONconfig.hpp>

class MainWindow : public GLFCameraWindow
{
public:
    MainWindow(const std::string &title,
               const Model *model,
               const int &launch_ray_type,
               const Camera &camera,
               const QuadLight &light,
               const float worldScale,
               const std::string &config_path);
    ~MainWindow();
    void applyConfig();

private:
    inline Camera get_camera() const
    {
        return Camera{cameraFrame.get_from(),
                      cameraFrame.get_at(),
                      cameraFrame.get_up(),
                      cameraFrame.get_frame()};
    }
    /*! rend one frame */
    void render() override;
    /*! pass the frame to opengl */
    void draw() override;
    void resize(const vec2i &newSize) override;

    /*! draw GUI layout */
    void draw_gui() override;

    vec2i fbSize;
    GLuint fbTexture{0};
    Renderer renderer;
    std::vector<uint32_t> pixels;

    LaunchParams *_params;  // Control parameters used by gui
    JSONconfig json_config; // JSON config file manip
};