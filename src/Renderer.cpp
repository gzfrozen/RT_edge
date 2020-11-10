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

#include <Renderer.hpp>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

// extern ptx code, same name as the file in src/cuda
extern "C" const unsigned char rayLaunch[];
extern "C" const unsigned char closestHit[];
extern "C" const unsigned char anyHit[];
extern "C" const unsigned char missHit[];
extern "C" const unsigned char callableProgram[];

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  TriangleMeshSBTData data;
};

/*! SBT record for a callable program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CallableRecord
{
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
Renderer::Renderer(const Model *model, const QuadLight &light)
    : model(model)
{
  initOptix();

  launchParams.light.origin = light.origin;
  launchParams.light.du = light.du;
  launchParams.light.dv = light.dv;
  launchParams.light.power = light.power;

  std::cout << "#osc: creating optix context ..." << std::endl;
  createContext();

  std::cout << "#osc: setting up module ..." << std::endl;
  createModule();

  std::cout << "#osc: creating raygen programs ..." << std::endl;
  createRaygenPrograms();
  std::cout << "#osc: creating miss programs ..." << std::endl;
  createMissPrograms();
  std::cout << "#osc: creating hitgroup programs ..." << std::endl;
  createHitgroupPrograms();
  std::cout << "#osc: creating callable programs ..." << std::endl;
  createCallablePrograms();
  std::cout << "#osc: building accel structure ..." << std::endl;
  launchParams.traversable = buildAccel();

  std::cout << "#osc: setting up optix pipeline ..." << std::endl;
  createPipeline();

  std::cout << "#osc: creating textures ..." << std::endl;
  createTextures();

  std::cout << "#osc: building SBT ..." << std::endl;
  buildSBT();

  launchParamsBuffer.alloc(sizeof(launchParams));
  std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

  std::cout << GDT_TERMINAL_GREEN;
  std::cout << "#osc: Optix 7 fully set up" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;
}

void Renderer::createTextures()
{
  int numTextures = (int)model->textures.size();

  textureArrays.resize(numTextures);
  textureObjects.resize(numTextures);

  for (int textureID = 0; textureID < numTextures; textureID++)
  {
    auto texture = model->textures[textureID];

    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t width = texture->resolution.x;
    int32_t height = texture->resolution.y;
    int32_t numComponents = 4;
    int32_t pitch = width * numComponents * sizeof(uint8_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t &pixelArray = textureArrays[textureID];
    CUDA_CHECK(MallocArray(&pixelArray,
                           &channel_desc,
                           width, height));

    CUDA_CHECK(Memcpy2DToArray(pixelArray,
                               /* offset */ 0, 0,
                               texture->pixel,
                               pitch, pitch, height,
                               cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = pixelArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    textureObjects[textureID] = cuda_tex;
  }
}

OptixTraversableHandle Renderer::buildAccel()
{
  const int numMeshes = (int)model->meshes.size();
  vertexBuffer.resize(numMeshes);
  normalBuffer.resize(numMeshes);
  texcoordBuffer.resize(numMeshes);
  indexBuffer.resize(numMeshes);

  OptixTraversableHandle asHandle{0};

  // ==================================================================
  // triangle inputs
  // ==================================================================
  std::vector<OptixBuildInput> triangleInput(numMeshes);
  std::vector<CUdeviceptr> d_vertices(numMeshes);
  std::vector<CUdeviceptr> d_indices(numMeshes);
  std::vector<uint32_t> triangleInputFlags(numMeshes);

  for (int meshID = 0; meshID < numMeshes; meshID++)
  {
    // upload the model to the device: the builder
    TriangleMesh &mesh = *model->meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
    indexBuffer[meshID].alloc_and_upload(mesh.index);
    if (!mesh.normal.empty())
      normalBuffer[meshID].alloc_and_upload(mesh.normal);
    if (!mesh.texcoord.empty())
      texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

    triangleInput[meshID] = {};
    triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    d_indices[meshID] = indexBuffer[meshID].d_pointer();

    triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
    triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
    triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

    triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
    triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
    triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

    triangleInputFlags[meshID] = 0;

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
  }

  // ==================================================================
  // BLAS setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                           &accelOptions,
                                           triangleInput.data(),
                                           numMeshes, // num_build_inputs
                                           &blasBufferSizes));

  // ==================================================================
  // prepare compaction
  // ==================================================================

  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();

  // ==================================================================
  // execute build (main stage)
  // ==================================================================

  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

  OPTIX_CHECK(optixAccelBuild(optixContext,
                              /* stream */ 0,
                              &accelOptions,
                              triangleInput.data(),
                              numMeshes,
                              tempBuffer.d_pointer(),
                              tempBuffer.sizeInBytes,

                              outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes,

                              &asHandle,

                              &emitDesc, 1));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // perform compaction
  // ==================================================================
  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize, 1);

  asBuffer.alloc(compactedSize);
  OPTIX_CHECK(optixAccelCompact(optixContext,
                                /*stream:*/ 0,
                                asHandle,
                                asBuffer.d_pointer(),
                                asBuffer.sizeInBytes,
                                &asHandle));
  CUDA_SYNC_CHECK();

  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================
  outputBuffer.free(); // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();

  return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void Renderer::initOptix()
{
  std::cout << "#osc: initializing optix..." << std::endl;

  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");
  std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());
  std::cout << GDT_TERMINAL_GREEN
            << "#osc: successfully initialized optix... yay!"
            << GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
  fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

/*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
void Renderer::createContext()
{
  // for this sample, do everything on one device
  const int deviceID = 0;
  CUDA_CHECK(SetDevice(deviceID));
  CUDA_CHECK(StreamCreate(&stream));

  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout << "#osc: running on device: " << deviceProps.name << std::endl;

  CUresult cuRes = cuCtxGetCurrent(&cudaContext);
  if (cuRes != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
      to use. we use modules from multiple .cu files, came from 
      multiple embedded ptx string which need to be loaded manually*/
void Renderer::createModule()
{
  moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  moduleCompileOptions.numBoundValues = 0;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
  pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  pipelineLinkOptions.maxTraceDepth = 1;

  // load each ptx code manually
  static std::unordered_map<std::string, const unsigned char *> ptxCode;
  ptxCode["rayLaunch"] = rayLaunch;
  ptxCode["closestHit"] = closestHit;
  ptxCode["anyHit"] = anyHit;
  ptxCode["missHit"] = missHit;
  ptxCode["callableProgram"] = callableProgram;

  for (auto [name, code] : ptxCode)
  {
    const std::string ptx = reinterpret_cast<const char *>(code);
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixModule m;
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptx.c_str(),
                                         ptx.size(),
                                         log, &sizeof_log,
                                         &m));
    if (sizeof_log > 1)
      PRINT(log);
    module[name] = m;
  }
}

/*! does all setup for the raygen program(s) we are going to use */
void Renderer::createRaygenPrograms()
{
  raygenPGs.resize(RENDERER_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module = module["rayLaunch"];

  // Fast renderer;
  pgDesc.raygen.entryFunctionName = "__raygen__fastRenderer";
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &raygenPGs[FAST]));
  if (sizeof_log > 1)
    PRINT(log);

  // Classic renderer;
  pgDesc.raygen.entryFunctionName = "__raygen__classicRenderer";
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &raygenPGs[CLASSIC]));
  if (sizeof_log > 1)
    PRINT(log);
}

/*! does all setup for the miss program(s) we are going to use */
void Renderer::createMissPrograms()
{
  missPGs.resize(RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module = module["missHit"];

  // ------------------------------------------------------------------
  // radiance rays
  // ------------------------------------------------------------------

  pgDesc.miss.entryFunctionName = "__miss__radiance";

  // OptixProgramGroup raypg;
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[RADIANCE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // ------------------------------------------------------------------
  // shadow rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__shadow";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[SHADOW_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // ------------------------------------------------------------------
  // phase detection rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__phase";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[PHASE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // ------------------------------------------------------------------
  // mono rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__mono";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[MONO_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // ------------------------------------------------------------------
  // edge detection rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__edge";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[EDGE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // ------------------------------------------------------------------
  // edge detection rays
  // ------------------------------------------------------------------
  pgDesc.miss.entryFunctionName = "__miss__classic";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &missPGs[CLASSIC_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void Renderer::createHitgroupPrograms()
{
  hitgroupPGs.resize(RAY_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH = module["closestHit"];
  pgDesc.hitgroup.moduleAH = module["anyHit"];

  // -------------------------------------------------------
  // radiance rays
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[RADIANCE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // shadow rays: technically we don't need this hit group,
  // since we just use the miss shader to check if we were not
  // in shadow
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[SHADOW_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // phase detection rays: rays used to detect phase change by
  // trace the distance between camer and hit point, no futher
  // shadow ray will be launched by this type of ray
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__phase";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__phase";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[PHASE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // mono rays: like radius rays but don't care about color
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__mono";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__mono";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[MONO_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // edge detection rays: rays used to detect edge, launched by mono rays
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__edge";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__edge";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[EDGE_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // Classic renderer rays: rays used in classic renderers
  // -------------------------------------------------------
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__classic";
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__classic";

  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &hitgroupPGs[CLASSIC_RAY_TYPE]));
  if (sizeof_log > 1)
    PRINT(log);
}

/*! does all setup for the callable program(s) we are going to use */
void Renderer::createCallablePrograms()
{
  callablePGs.resize(RENDERER_TYPE_COUNT);

  char log[2048];
  size_t sizeof_log = sizeof(log);

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc = {};
  pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;

  // -------------------------------------------------------
  // fast launch: fast edge renderer
  // -------------------------------------------------------
  pgDesc.callables.moduleCC = module["callableProgram"];
  pgDesc.callables.entryFunctionNameCC = "__continuation_callable__fast_launch";

  // OptixProgramGroup raypg;
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &callablePGs[FAST]));
  if (sizeof_log > 1)
    PRINT(log);

  // -------------------------------------------------------
  // classic launch: classic edge renderer
  // -------------------------------------------------------
  pgDesc.callables.moduleCC = module["callableProgram"];
  pgDesc.callables.entryFunctionNameCC = "__continuation_callable__classic_launch";

  // OptixProgramGroup raypg;
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log, &sizeof_log,
                                      &callablePGs[CLASSIC]));
  if (sizeof_log > 1)
    PRINT(log);
}

/*! assembles the full pipeline of all programs */
void Renderer::createPipeline()
{
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs)
    programGroups.push_back(pg);
  for (auto pg : missPGs)
    programGroups.push_back(pg);
  for (auto pg : hitgroupPGs)
    programGroups.push_back(pg);
  for (auto pg : callablePGs)
    programGroups.push_back(pg);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  PING;
  PRINT(programGroups.size());
  OPTIX_CHECK(optixPipelineCreate(optixContext,
                                  &pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups.data(),
                                  (int)programGroups.size(),
                                  log, &sizeof_log,
                                  &pipeline));
  if (sizeof_log > 1)
    PRINT(log);

  // OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
  //                                       pipeline,
  //                                       /* [in] The direct stack size requirement for direct
  //                   callables invoked from IS or AH. */
  //                                       2 * 1024,
  //                                       /* [in] The direct stack size requirement for direct
  //                   callables invoked from RG, MS, or CH.  */
  //                                       2 * 1024,
  //                                       /* [in] The continuation stack requirement. */
  //                                       2 * 1024,
  //                                       /* [in] The maximum depth of a traversable graph
  //                   passed to trace. */
  //                                       1));
  // if (sizeof_log > 1)
  //   PRINT(log);
}

/*! constructs the shader binding table */
void Renderer::buildSBT()
{
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i = 0; i < raygenPGs.size(); i++)
  {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back(rec);
  }
  raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (int i = 0; i < missPGs.size(); i++)
  {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
    rec.data = nullptr; /* for now ... */
    missRecords.push_back(rec);
  }
  missRecordsBuffer.alloc_and_upload(missRecords);
  sbt.missRecordBase = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  int numObjects = (int)model->meshes.size();
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int meshID = 0; meshID < numObjects; meshID++)
  {
    auto mesh = model->meshes[meshID];
    for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
    {
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
      rec.data.color = mesh->diffuse;
      if (mesh->diffuseTextureID >= 0)
      {
        rec.data.hasTexture = true;
        rec.data.texture = textureObjects[mesh->diffuseTextureID];
      }
      else
      {
        rec.data.hasTexture = false;
      }
      rec.data.geometryID = mesh->geometryID;
      rec.data.index = (vec3i *)indexBuffer[meshID].d_pointer();
      rec.data.vertex = (vec3f *)vertexBuffer[meshID].d_pointer();
      rec.data.normal = (vec3f *)normalBuffer[meshID].d_pointer();
      rec.data.texcoord = (vec2f *)texcoordBuffer[meshID].d_pointer();
      hitgroupRecords.push_back(rec);
    }
  }
  hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
  sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

  // ------------------------------------------------------------------
  // build callable records
  // ------------------------------------------------------------------
  std::vector<CallableRecord> callableRecords;
  for (int i = 0; i < callablePGs.size(); i++)
  {
    CallableRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(callablePGs[i], &rec));
    rec.data = nullptr; /* for now ... */
    callableRecords.push_back(rec);
  }
  callableRecordsBuffer.alloc_and_upload(callableRecords);
  sbt.callablesRecordBase = callableRecordsBuffer.d_pointer();
  sbt.callablesRecordStrideInBytes = sizeof(CallableRecord);
  sbt.callablesRecordCount = (int)callableRecords.size();
}

/*! render one frame */
void Renderer::render()
{
  // sanity check: make sure we launch only after first resize is
  // already done:
  if (launchParams.frame.size.x == 0)
    return;

  launchParamsBuffer.upload(&launchParams, 1);
  launchParams.frame.accumID++;

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: multiple pipelines not working*/
                          pipeline, stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.d_pointer(),
                          launchParamsBuffer.sizeInBytes,
                          &sbt,
                          /*! dimensions of the launch: */
                          launchParams.frame.size.x,
                          launchParams.frame.size.y,
                          1));
  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void Renderer::setCamera(const Camera &camera)
{
  launchParams.camera.camera_type = PINHOLE;
  lastSetCamera = camera;
  launchParams.camera.position = camera.from;
  launchParams.camera.direction = normalize(camera.at - camera.from);
  const float cosFovy = 0.66f;
  const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
  launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
  launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                                           launchParams.camera.direction));
}

/*! set sphere camera to render with */
void Renderer::setEnvCamera(const Camera &camera)
{
  launchParams.camera.camera_type = ENV;
  lastSetCamera = camera;
  launchParams.camera.position = camera.from;
  const vec3f direction = camera.at - camera.from;
  launchParams.camera.direction = normalize(direction);
  const float unit_theta = 2 * M_PI / float(launchParams.frame.size.x);
  const float unit_phi = M_PI / float(launchParams.frame.size.y);
  launchParams.camera.horizontal = {0.f, unit_theta, 0.f};
  launchParams.camera.vertical = {0.f, 0.f, unit_phi};
  launchParams.camera.matrix = linear3f{-camera.matrix.vz,
                                        -camera.matrix.vx,
                                        camera.matrix.vy};
}

/*! set ray stencil used in classic renderer */
void Renderer::setRayStencil()
{
  float &h = launchParams.classic.RAY_STENCIL_RADIUS;
  int &N = launchParams.classic.RAY_STENCIL_QUALITY.x;
  int &n = launchParams.classic.RAY_STENCIL_QUALITY.y;

  // parameter check
  assert(h > 0.f);
  assert(N > 0 && N < 9);
  assert(n > 0 && n % 4 == 0);

  float temp_r;
  int temp_n{n};
  float theta;
  int index{0};

  for (int i = 0; i < N; i++)
  {
    temp_r = h / N * (i + 1);
    for (int j = 0; j < temp_n; j++)
    {
      theta = 2 * M_PI * j / temp_n;
      // calculate screen space offset position
      launchParams.classic.ray_stencil[index + j] = vec2f(cosf(theta) * temp_r, sinf(theta) * temp_r);
    }
    index += temp_n;
    temp_n *= 2;
  }

  launchParams.classic.stencil_length = index;
}

/*! set ray type used in __raygen__ */
void Renderer::setLaunchRayType(const int &launch_ray_type)
{
  launchParams.parameters.LAUNCH_RAY_TYPE = launch_ray_type;
}

/*! set renderer type used in __raygen__ */
void Renderer::setRendererType(const int &renderer_type)
{
  launchParams.parameters.RENDERER_TYPE = renderer_type;
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer() + sizeof(MissRecord) * renderer_type;
}

/*! return the pointer of launch parameters */
LaunchParams *Renderer::getLaunchParams()
{
  return &launchParams;
}

/*! resize frame buffer to given resolution */
void Renderer::resize(const vec2i &newSize)
{
  // if window minimized
  if (newSize.x == 0 || newSize.y == 0)
    return;
  // resize our cuda frame buffer
  colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.frame.size = newSize;
  launchParams.frame.colorBuffer = (uint32_t *)colorBuffer.d_pointer();

  // and re-set the camera, since aspect may have changed
  setCamera(lastSetCamera);
}

/*! download the rendered color buffer */
void Renderer::downloadPixels(uint32_t h_pixels[])
{
  colorBuffer.download(h_pixels,
                       launchParams.frame.size.x * launchParams.frame.size.y);
}
