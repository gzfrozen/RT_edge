#include <MainWindow.hpp>
#include <GL/gl.h>

// ImGui framework
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl2.h>

// Gui control parameters
#include <LaunchParams.hpp>

MainWindow::MainWindow(const std::string &title,
                       const Model *model,
                       const int &launch_ray_type,
                       const Camera &camera,
                       const QuadLight &light,
                       const float worldScale)
    : GLFCameraWindow(title, launch_ray_type, camera.from, camera.at, camera.up, worldScale),
      sample(model, light)
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

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    ImGui_ImplOpenGL2_Init();
}

MainWindow::~MainWindow()
{
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    GLFWindow::~GLFWindow();
}

void MainWindow::before_render()
{
    if (cameraFrameManip)
        cameraFrameManip->move_wsad();
}

void MainWindow::render()
{
    if (cameraFrame.modified)
    {
        sample.setLaunchRayType(ray_type);
        int cameraType = cameraFrame.get_camera_type();
        if (cameraType == PINHOLE)
        {
            sample.setCamera(get_camera());
        }
        else if (cameraType == ENV)
        {
            sample.setEnvCamera(get_camera());
        }
        cameraFrame.modified = false;
    }
    sample.render();
}

void MainWindow::draw()
{
    sample.downloadPixels(pixels.data());
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
    sample.resize(newSize);
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
            static LaunchParams &params = sample.getLaunchParams();
            ImGui::Begin("Settings"); // Create a window called "Hello, world!" and append into it.
            ImGui::SetWindowFontScale(xscale);

            ImGui::SliderFloat("float", &params.parameters.MAX_EDGE_DISTANCE, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::InputFloat("test", &params.parameters.MAX_EDGE_DISTANCE);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }
    }

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}