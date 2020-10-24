#include <JSONconfig.hpp>
#include <iomanip>

JSONconfig::JSONconfig()
    : _params(nullptr), path("config.json")
{
}

JSONconfig::JSONconfig(const std::string &path, LaunchParams *const _params, const Camera &camera)
    : path(path), _params(_params), config(json())
{
    generateConfig(camera);
}

JSONconfig &JSONconfig::operator=(const JSONconfig &other)
{
    path = other.path;
    _params = other._params;
    config = other.config;
    return *this;
}

JSONconfig &JSONconfig::operator=(JSONconfig &&other)
{
    if (this != &other)
    {
        path = (std::move)(other.path);
        _params = other._params;
        config = (std::move)(other.config);
        other._params = nullptr;
    }
    return *this;
}

void JSONconfig::readFile()
{
    std::ifstream file(path);
    if (file.fail())
    {
        std::cout << GDT_TERMINAL_GREEN << "Creating new config file: " << GDT_TERMINAL_DEFAULT << path << std::endl;
        saveFile();
    }
    else
    {
        config.clear();
        file >> config;
    }
}

int JSONconfig::saveFile() const
{
    if (config.empty())
    {
        std::cout << GDT_TERMINAL_RED << "Config is empty." << GDT_TERMINAL_DEFAULT << std::endl;
        return -1;
    }
    std::ofstream file(path);
    if (file.fail())
    {
        std::cout << GDT_TERMINAL_RED << "Failed to open file." << GDT_TERMINAL_DEFAULT << std::endl;
        return -1;
    }
    else
    {
        file << std::setw(4) << config << std::endl;
        std::cout << GDT_TERMINAL_GREEN << "Config saved in: " << GDT_TERMINAL_DEFAULT << path << std::endl;
        return 0;
    }
}

void JSONconfig::generateConfig(const Camera &camera)
{
    // Parameters which will be returned as a Camera type
    config["camera"]["from"]["x"] = camera.from.x;
    config["camera"]["from"]["y"] = camera.from.y;
    config["camera"]["from"]["z"] = camera.from.z;
    config["camera"]["at"]["x"] = camera.at.x;
    config["camera"]["at"]["y"] = camera.at.y;
    config["camera"]["at"]["z"] = camera.at.z;
    config["camera"]["up"]["x"] = camera.up.x;
    config["camera"]["up"]["y"] = camera.up.y;
    config["camera"]["up"]["z"] = camera.up.z;

    // Parameters which will be returned as an enum(int) type
    config["parameters"]["LAUNCH_RAY_TYPE"] = _params->parameters.LAUNCH_RAY_TYPE;

    // Parameters which are directly updated into memory
    config["parameters"]["NUM_LIGHT_SAMPLES"] = _params->parameters.NUM_LIGHT_SAMPLES;
    config["parameters"]["NUM_PIXEL_SAMPLES"] = _params->parameters.NUM_PIXEL_SAMPLES;
    config["parameters"]["WAVE_LENGTH"] = _params->parameters.WAVE_LENGTH;
    config["parameters"]["EDGE_DETECTION_DEPTH"] = _params->parameters.EDGE_DETECTION_DEPTH;
    config["parameters"]["MAX_EDGE_DISTANCE"] = _params->parameters.MAX_EDGE_DISTANCE;
    config["parameters"]["MAX_EDGE_ANGLE"] = _params->parameters.MAX_EDGE_ANGLE;
}

void JSONconfig::applyConfig() const
{
    // Parameters which are directly updated into memory
    // _params->parameters.LAUNCH_RAY_TYPE = config["parameters"]["LAUNCH_RAY_TYPE"];
    _params->parameters.NUM_LIGHT_SAMPLES = config["parameters"]["NUM_LIGHT_SAMPLES"];
    _params->parameters.NUM_PIXEL_SAMPLES = config["parameters"]["NUM_PIXEL_SAMPLES"];
    _params->parameters.WAVE_LENGTH = config["parameters"]["WAVE_LENGTH"];
    _params->parameters.EDGE_DETECTION_DEPTH = config["parameters"]["EDGE_DETECTION_DEPTH"];
    _params->parameters.MAX_EDGE_DISTANCE = config["parameters"]["MAX_EDGE_DISTANCE"];
    _params->parameters.MAX_EDGE_ANGLE = config["parameters"]["MAX_EDGE_ANGLE"];
}

Camera JSONconfig::returnCamera() const
{
    Camera camera;
    camera.from = vec3f{
        config["camera"]["from"]["x"],
        config["camera"]["from"]["y"],
        config["camera"]["from"]["z"]};
    camera.at = vec3f{
        config["camera"]["at"]["x"],
        config["camera"]["at"]["y"],
        config["camera"]["at"]["z"]};
    camera.up = vec3f{
        config["camera"]["up"]["x"],
        config["camera"]["up"]["y"],
        config["camera"]["up"]["z"]};
    return camera;
}

int JSONconfig::returnRayType() const
{
    return config["parameters"]["LAUNCH_RAY_TYPE"];
}