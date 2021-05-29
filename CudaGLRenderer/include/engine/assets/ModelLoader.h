#pragma once

#include "Model.h"
#include <tiny_gltf/tiny_gltf.h>

namespace utad
{
	namespace gltf
	{
		using namespace tinygltf;
	}

	struct ModelInfo
	{
		gltf::Model* model;
		Map<int, Buffer*>* buffers;
		VertexArray* vertexArray;
	};

	class ModelLoader
	{
	public:
		ModelLoader();
		~ModelLoader();
		Model* load(const String& name, const String& path);
	private:
		void loadNode(ModelInfo& info, gltf::Node& node, ModelNode& result);
		void createBuffers(ModelInfo& info);
		void loadMesh(ModelInfo& info, gltf::Node& node, ModelNode& result);
		void loadMaterial(ModelInfo& info, gltf::Node& node, ModelNode& result);
		Buffer* createGLBuffer(uint binding, const ModelInfo& info, const gltf::BufferView& bufferView, const gltf::Buffer& buffer);
	};

}