#include <MainWindow.hpp>
#include <GL/gl.h>

// ImGui framework
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl2.h>

// Gui control parameters
#include <LaunchParams.hpp>

// JSON config
#include <JSONconfig.hpp>

// Capture screen
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

MainWindow::MainWindow(const std::string &title,
                       const Model *model,
                       const Camera &camera,
                       const QuadLight &light,
                       const float worldScale,
                       const std::string &config_path)
    : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
      renderer(model, light)
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();
    //ImGuiStyle style = ImGui::GetStyle();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    ImGui_ImplOpenGL2_Init();

    // Get control parameters
    _params = renderer.getLaunchParams();

    // Read configure file on start
    json_config = JSONconfig(config_path, _params, get_camera());
    applyConfig();
}

MainWindow::~MainWindow()
{
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    GLFWindow::~GLFWindow();
}

void MainWindow::render()
{
    if (cameraFrame.modified)
    {
        renderer.setRendererType(renderer_type);
        renderer.setLaunchRayType(ray_type);
        if (_params->parameters.RENDERER_TYPE == CLASSIC)
        {
            renderer.setRayStencil();
        }
        int cameraType = cameraFrame.get_camera_type();
        if (cameraType == PINHOLE)
        {
            renderer.setCamera(get_camera());
        }
        else if (cameraType == ENV)
        {
            renderer.setEnvCamera(get_camera());
        }
        cameraFrame.modified = false;
    }
    renderer.render();
}

void MainWindow::draw()
{
    renderer.downloadPixels(pixels.data());
    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                 texelType, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();
}

void MainWindow::resize(const vec2i &newSize)
{
    fbSize = newSize;
    renderer.resize(newSize);
    pixels.resize(newSize.x * newSize.y);
}

void MainWindow::draw_gui()
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        if (ui_on)
        {
            ImGui::Begin("Settings", &ui_on, ImGuiWindowFlags_HorizontalScrollbar); // Create a window
            ImGui::SetWindowFontScale(xscale);

            if (renderer_type == FAST || _params->parameters.RENDERER_TYPE == MIXED)
            {
                if (_params->parameters.LAUNCH_RAY_TYPE == RADIANCE_RAY_TYPE || _params->parameters.RENDERER_TYPE == MIXED)
                {
                    ImGui::PushItemWidth(200.f * xscale);
                    ImGui::SliderFloat3("Light Direction", reinterpret_cast<float *>(&_params->light.origin), -5000.0f, 5000.0f, "%.1f");
                }
                ImGui::PushItemWidth(70.f * xscale);
                ImGui::InputInt("Number of light samples", &_params->parameters.NUM_LIGHT_SAMPLES, 1, 5);
                ImGui::SameLine(0.f, 20.f * xscale);
                ImGui::InputInt("Number of pixel samples", &_params->parameters.NUM_PIXEL_SAMPLES, 1, 5);
                if (_params->parameters.LAUNCH_RAY_TYPE == PHASE_RAY_TYPE)
                {
                    ImGui::PushItemWidth(100.f * xscale);
                    ImGui::SliderFloat("Wave Length", &_params->parameters.WAVE_LENGTH, 0.0f, 1000.0f);
                }
                else if (_params->parameters.LAUNCH_RAY_TYPE == MONO_RAY_TYPE || _params->parameters.RENDERER_TYPE == MIXED)
                {
                    ImGui::PushItemWidth(100.f * xscale);
                    ImGui::Checkbox("Edge on angle over PI", &_params->parameters.OVER_PI_EDGE);
                    ImGui::SliderFloat("Edge detection depth", &_params->parameters.EDGE_DETECTION_DEPTH, 1e-3f, 5.f, "%5.3e");
                    ImGui::SliderFloat("Max edge distance", &_params->parameters.MAX_EDGE_DISTANCE, 0.0f, 1.0f, "%5.3e");
                    ImGui::SliderFloat("Min edge angle", &_params->parameters.MIN_EDGE_ANGLE, 0.0f, _params->parameters.MAX_EDGE_ANGLE, "%.3f");
                    ImGui::SliderFloat("Max edge angle", &_params->parameters.MAX_EDGE_ANGLE, 0.0f, M_PI, "%.3f");
                }
            }
            else if (renderer_type == CLASSIC)
            {
                ImGui::PushItemWidth(100.f * xscale);
                if (ImGui::SliderFloat("Ray stencil radius", &_params->classic.RAY_STENCIL_RADIUS, 0.0f, 0.2f, "%5.3e"))
                    cameraFrame.modified = true;
                ImGui::SameLine(0.f, 20.f * xscale);
                ImGui::PushItemWidth(70.f * xscale);
                if (ImGui::InputInt2("Ray stencil quality", reinterpret_cast<int *>(&_params->classic.RAY_STENCIL_QUALITY)))
                    cameraFrame.modified = true;

                ImGui::PushItemWidth(100.f * xscale);
                ImGui::SliderFloat("Crease edge angle threshold", &_params->classic.NORMAL_CHANGE_THRESHOLD, 0.0f, M_PI);
                ImGui::SliderFloat("Self-occluding silhouette distance threshold", &_params->classic.DISTANCE_CHANGE_THRESHOLD, 0.0f, 1.f);
            }

            ImGui::NewLine();
            if (ImGui::Button("Load")) // Buttons return true when clicked (most widgets return true when edited/activated)
            {
                applyConfig();
            }
            ImGui::SameLine(0.f, 10.f * xscale);
            if (ImGui::Button("Save"))
            {
                json_config.generateConfig(get_camera());
                json_config.saveFile();
            }
            ImGui::SameLine(0.f, 10.f * xscale);
            if (ImGui::Button("Capture"))
            {
                static int capID{0};
                std::ostringstream path;
                path << "cap" << capID << ".png";
                capture(path.str());
                capID++;
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS), under %d * %d",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate,
                        fbSize.x,
                        fbSize.y);
            ImGui::End();
        }
    }

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

void MainWindow::applyConfig()
{
    json_config.readFile();
    json_config.applyConfig();
    Camera configCamera = json_config.returnCamera();
    cameraFrame.setOrientation(configCamera.from, configCamera.at, configCamera.up);
    renderer_type = json_config.returnRendererType();
    ray_type = json_config.returnRayType();
}

void MainWindow::capture(const std::string &path)
{
    /* iw - actually, it seems that stbi writes the pictures
         mirrored along the y axis - mirror them here */
    for (int y = 0; y < fbSize.y / 2; y++)
    {
        uint32_t *line_y = pixels.data() + y * fbSize.x;
        uint32_t *mirrored_y = pixels.data() + (fbSize.y - 1 - y) * fbSize.x;
        for (int x = 0; x < fbSize.x; x++)
        {
            std::swap(line_y[x], mirrored_y[x]);
        }
    }
    stbi_write_png(path.c_str(), fbSize.x, fbSize.y, 4, pixels.data(), fbSize.x * sizeof(uint32_t));
    std::cout << GDT_TERMINAL_GREEN << "Screenshot saved as " << path << GDT_TERMINAL_DEFAULT << std::endl;
}