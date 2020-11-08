#pragma once

#include <fstream>
#include <json.hpp>

#include <LaunchParams.hpp>
#include <Renderer.hpp>

using json = nlohmann::json;

class JSONconfig
{
public:
    JSONconfig();
    JSONconfig(const std::string &path, LaunchParams *const _params, const Camera &camera);
    JSONconfig &operator=(const JSONconfig &other);
    JSONconfig &operator=(JSONconfig &&other);

    void readFile();
    int saveFile() const;
    void generateConfig(const Camera &camera); // Generate (json) config member, a json object
    void applyConfig() const;                  //Set parameters which are directly updated into memory
    Camera returnCamera() const;               //Return a Camera type to used in undirectly camera update
    int returnRayType() const;                 //Return the ray type to used in undirectly camera update
    int returnRendererType() const;            //Return the renderer type to used in undirectly camera update

private:
    std::string path{"config.json"}; //JSON config file path
    LaunchParams *_params;           // Pointer of control parameters
    json config;                     // JSON object
};