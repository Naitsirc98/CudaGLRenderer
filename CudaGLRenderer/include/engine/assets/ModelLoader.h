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
		Map<int, BufferView>* buffers;
		ArrayList<Texture2D*>* textures;
	};

	class ModelLoader
	{
	private:
		Map<String, VertexAttrib> m_VertexAttribsByName;
		bool m_DebugMode{false};
	public:
		ModelLoader();
		~ModelLoader();
		Model* load(const String& name, const String& path);
		bool debugMode() const;
		void debugMode(bool debugMode);
	private:
		void loadNode(ModelInfo& info, gltf::Node& node, ModelNode& result);
		void loadBuffers(ModelInfo& info);
		void loadTransformation(const ArrayList<double>& matrix, ModelNode& result);
		void loadMesh(ModelInfo& info, gltf::Node& node, ModelNode& result);
		void setupVertexBuffers(ModelInfo& info, const gltf::Mesh& mesh, VertexArray* vertexArray);
		bool getVertexAttrib(const String& attribName, VertexAttrib& attribute) const;
		void loadMaterial(ModelInfo& info, gltf::Node& node, ModelNode& result);
		Buffer* createGLBuffer(const gltf::BufferView& bufferView, const gltf::Buffer& buffer);
		void loadTextures(ModelInfo& info);
	};

}