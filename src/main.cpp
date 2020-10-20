// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// our helper library for window handling
#include <GL/gl.h>
#include <MainWindow.hpp>

/*! main entry point to this example - initially optix, print hello
    world, then exit */
extern "C" int main(int ac, char **av)
{
    try
    {
        Model *model = loadOBJ(_OBJ_FILE);
        // set launch ray type: RADIANCE_RAY_TYPE or PHASE_RAY_TYPE
        int launch_ray_type = RADIANCE_RAY_TYPE;
        // set launch camera position
        Camera camera = {/*from*/ vec3f(-1293.07f, 154.681f, -0.7304f),
                         /* at */ model->bounds.center() - vec3f(0, 400, 0),
                         /* up */ vec3f(0.f, 1.f, 0.f)};

        // some simple, hard-coded light ... obviously, only works for sponza
        const float light_size = 200.f;
        QuadLight light = {/* origin */ vec3f(-1000 - light_size, 800, -light_size),
                           /* edge 1 */ vec3f(2.f * light_size, 0, 0),
                           /* edge 2 */ vec3f(0, 0, 2.f * light_size),
                           /* power */ vec3f(3000000.f)};

        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = length(model->bounds.span());

        // JSON config file path
        const std::string config_path = "config.json";

        MainWindow *window = new MainWindow("RT Edge",
                                            model, launch_ray_type, camera, light, worldScale, config_path);
        window->run();
    }
    catch (std::runtime_error &e)
    {
        std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                  << GDT_TERMINAL_DEFAULT << std::endl;
        std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your RT_edge/models directory?" << std::endl;
        exit(1);
    }
    return 0;
}
