#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_USE_CPP14
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf/tiny_gltf.h>
#include "engine/assets/ModelLoader.h"

namespace utad
{
	namespace gltf 
	{
		using namespace tinygltf;
	}

	static MeshID g_NextMeshID = 1;

	ModelLoader::ModelLoader()
	{
	}

	ModelLoader::~ModelLoader()
	{
	}

	Model* ModelLoader::load(const String& name, const String& path)
	{
		gltf::Model gltfModel;
		gltf::TinyGLTF loader;
		String error;
		String warning;

		bool success = loader.LoadASCIIFromFile(&gltfModel, &error, &warning, path);

		if (!success) throw UTAD_EXCEPTION(String("Failed to load GLTF2.0 model ").append(path).append(": ").append(error));
		if (!error.empty()) std::cout << "Error in GLTF2.0 model " << path << " : " << error << std::endl;
		if (!warning.empty()) std::cout << "Warning in GLTF2.0 model " << path << " : " << error << std::endl;

		return nullptr;
	}
}