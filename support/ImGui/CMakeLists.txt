add_library(ImGui
imconfig.h
imgui_demo.cpp
imgui_draw.cpp
imgui_impl_glfw.cpp
imgui_impl_glfw.h
imgui_impl_opengl2.cpp
imgui_impl_opengl2.h
imgui_internal.h
imgui_widgets.cpp
imgui.cpp
imgui.h
imstb_rectpack.h
imstb_textedit.h
imstb_truetype.h
)

target_precompile_headers(ImGui PUBLIC 
    <imgui.h>
    <imgui_impl_glfw.h>
    <imgui_impl_opengl2.h>
    )