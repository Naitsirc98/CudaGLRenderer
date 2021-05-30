#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include "engine/assets/ModelLoader.h"
#include "engine/assets/AssetsManager.h"

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
		gltf::Model model;
		gltf::TinyGLTF loader;
		String error;
		String warning;

		bool success = loader.LoadASCIIFromFile(&model, &error, &warning, path);

		if (!success) throw UTAD_EXCEPTION(String("Failed to load GLTF2.0 model ").append(path).append(": ").append(error));
		if (!error.empty()) std::cout << "Error in GLTF2.0 model " << path << " : " << error << std::endl;
		if (!warning.empty()) std::cout << "Warning in GLTF2.0 model " << path << " : " << error << std::endl;

		Model* result = new Model();
		VertexArray* vertexArray = new VertexArray();
		vertexArray->bind();
		{
			Map<int, Buffer*> buffers;

			ModelInfo info;
			info.model = &model;
			info.buffers = &buffers;
			info.vertexArray = vertexArray;

			gltf::Scene& scene = model.scenes[model.defaultScene];
			for (int nodeIndex : scene.nodes)
			{
				loadNode(info, model.nodes[nodeIndex], *result->createNode());
			}
		}
		vertexArray->unbind();
		vertexArray->destroyVertexBuffers();

		return result;
	}

	void ModelLoader::loadNode(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		result.m_Name = node.name;
		createBuffers(info);
		loadMesh(info, node, result);
		loadMaterial(info, node, result);
	}

	void ModelLoader::createBuffers(ModelInfo& info)
	{
		gltf::Model& model = *info.model;
		Map<int, Buffer*>& buffers = *info.buffers;
		
		for (size_t i = 0; i < model.bufferViews.size(); ++i)
		{
			const gltf::BufferView& bufferView = model.bufferViews[i];
			if (bufferView.target == 0) continue;
			const gltf::Buffer& buffer = model.buffers[bufferView.buffer];
			buffers[i] = createGLBuffer(i, info, bufferView, buffer);
		}
	}

	void ModelLoader::loadMesh(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		gltf::Model& model = *info.model;
		Map<int, Buffer*>& buffers = *info.buffers;
		VertexArray& vao = *info.vertexArray;

		gltf::Mesh& mesh = model.meshes[node.mesh];
		gltf::Primitive primitive = mesh.primitives[0]; // Only supports 1 primitive per mesh for now
		gltf::Accessor& indexAccessor = model.accessors[primitive.indices];

		for (const auto [attribName, accessorIndex] : primitive.attributes)
		{
			gltf::Accessor& accessor = model.accessors[accessorIndex];
			const uint stride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);

			int count = 1;
			if (accessor.type != TINYGLTF_TYPE_SCALAR) count = accessor.type;

			int binding = accessor.bufferView;
			VertexBuffer* buffer = static_cast<VertexBuffer*>(buffers[binding]);
			vao.addVertexBuffer(binding, buffer, stride);

			int location = -1;
			if (attribName == "POSITION") location = 0;
			else if (attribName == "NORMAL") location = 1;
			else if (attribName == "TEXCOORD_0") location = 2;

			VertexAttrib attrib;
			attrib.count = count;
			attrib.type = accessor.componentType;

			if (location >= 0)
				vao.setVertexAttrib(binding, attrib, location, model.bufferViews[binding].byteOffset);
			else
				std::cout << "Missing vaa: " << attribName << std::endl;
		}
	}

	template<typename T>
	static Color toColor(const ArrayList<T>& values)
	{
		Color color;
		color.r = values.size() > 0 ? static_cast<float>(values[0]) : 0.0f;
		color.g = values.size() > 1 ? static_cast<float>(values[1]) : 0.0f;
		color.b = values.size() > 2 ? static_cast<float>(values[2]) : 0.0f;
		color.a = values.size() > 3 ? static_cast<float>(values[3]) : 1.0f;
		return color;
	}

	void ModelLoader::loadMaterial(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		gltf::Model& model = *info.model;
		gltf::Mesh& mesh = model.meshes[node.mesh];
		gltf::Material& mat = model.materials[mesh.primitives[0].material];
		gltf::PbrMetallicRoughness& pbr = mat.pbrMetallicRoughness;

		Material* material = Assets::getMaterial(mesh.name);
		if (material == nullptr) material = Assets::createMaterial(mesh.name);

		material->emissiveColor(toColor(mat.emissiveFactor));
		material->albedo(toColor(pbr.baseColorFactor));
	}

	Buffer* ModelLoader::createGLBuffer(uint binding, const ModelInfo& info, const gltf::BufferView& bufferView, const gltf::Buffer& buffer)
	{
		Buffer* result = Buffer::create(bufferView.target);

		BufferAllocInfo allocInfo = {};
		allocInfo.size = bufferView.byteLength;
		allocInfo.data = const_cast<unsigned char*>(buffer.data.data() + bufferView.byteOffset);

		result->allocate(allocInfo);

		return result;
	}
}