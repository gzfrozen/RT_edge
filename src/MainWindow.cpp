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

MainWindow::MainWindow(const std::string &title,
                       const Model *model,
                       const int &launch_ray_type,
                       const Camera &camera,
                       const QuadLight &light,
                       const float worldScale,
                       const std::string &config_path)
    : GLFCameraWindow(title, launch_ray_type, camera.from, camera.at, camera.up, worldScale),
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
        renderer.setLaunchRayType(ray_type);
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
            ImGui::Begin("Settings", &ui_on); // Create a window called "Hello, world!" and append into it.
            ImGui::SetWindowFontScale(xscale);

            ImGui::InputInt("Number of light samples", &_params->parameters.NUM_LIGHT_SAMPLES, 1, 5);
            ImGui::InputInt("Number of pixel samples", &_params->parameters.NUM_PIXEL_SAMPLES, 1, 5);

            ImGui::SliderFloat("Wave Length", &_params->parameters.WAVE_LENGTH, 0.0f, 1000.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::SliderFloat("Edge detection depth", &_params->parameters.EDGE_DETECTION_DEPTH, 0.0f, 1e-3f, "%e");
            ImGui::SliderFloat("Max edge distance", &_params->parameters.MAX_EDGE_DISTANCE, 0.0f, 1.0f);
            ImGui::SliderFloat("Max edge angle", &_params->parameters.MAX_EDGE_ANGLE, 0.0f, M_PI);

            ImGui::NewLine();
            if (ImGui::Button("Load")) // Buttons return true when clicked (most widgets return true when edited/activated)
            {
                applyConfig();
            }
            ImGui::SameLine();
            // ImGui::SameLine();
            if (ImGui::Button("Save"))
            {
                json_config.generateConfig(get_camera());
                json_config.saveFile();
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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
    ray_type = json_config.returnRayType();
}