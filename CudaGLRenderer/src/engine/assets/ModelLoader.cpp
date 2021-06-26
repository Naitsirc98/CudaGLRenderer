#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include "engine/assets/ModelLoader.h"
#include "engine/assets/AssetsManager.h"
#include "engine/assets/Image.h"
#include <chrono>

namespace utad
{
	using namespace std::chrono;

	namespace gltf 
	{
		using namespace tinygltf;
	}

	static MeshID g_NextMeshID = 1;

	static const char* targetToStr(GLenum target)
	{
		switch (target)
		{
			case GL_ARRAY_BUFFER: return "Vertex Buffer";
			case GL_ELEMENT_ARRAY_BUFFER: return "Index Buffer";
		}
		return "Other";
	}

	ModelLoader::ModelLoader()
	{
		m_VertexAttribsByName["POSITION"] = VertexAttrib::Position;
		m_VertexAttribsByName["NORMAL"] = VertexAttrib::Normal;
		m_VertexAttribsByName["TEXCOORD_0"] = VertexAttrib::TexCoords;
		m_VertexAttribsByName["TANGENT"] = VertexAttrib::Tangent;
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
		if (m_DebugMode && !error.empty()) std::cout << "Error in GLTF2.0 model " << path << " : " << error << std::endl;
		if (m_DebugMode && !warning.empty()) std::cout << "Warning in GLTF2.0 model " << path << " : " << error << std::endl;

		if (m_DebugMode) std::cout << "Parsing model " << name << "..." << std::endl;

		auto start = steady_clock::now();

		Model* result = new Model();
		Map<int, BufferView> buffers;
		ArrayList<Texture2D*> textures;

		ModelInfo info;
		info.model = &model;
		info.buffers = &buffers;
		info.textures = &textures;

		loadBuffers(info);
		loadTextures(info);

		gltf::Scene& scene = model.scenes[model.defaultScene];
		for (int nodeIndex : scene.nodes)
		{
			loadNode(info, model.nodes[nodeIndex], *result->createNode());
		}

		auto end = steady_clock::now();
		auto time = duration_cast<milliseconds>(end - start).count();

		if (m_DebugMode) std::cout << "Model " << name << " loaded in " << time << " milliseconds" << std::endl;

		return result;
	}

	bool ModelLoader::debugMode() const
	{
		return m_DebugMode;
	}

	void ModelLoader::debugMode(bool debugMode)
	{
		m_DebugMode = debugMode;
	}

	void ModelLoader::loadNode(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		if (m_DebugMode) std::cout << "\tLoading node " << node.name << std::endl;

		result.m_Name = node.name;

		loadTransformation(node.matrix, result);
		loadMesh(info, node, result);
		loadMaterial(info, node, result);

		for (int childIndex : node.children)
		{
			gltf::Node& child = info.model->nodes[childIndex];
			loadNode(info, child, *result.createChild());
		}
	}

	void ModelLoader::loadTransformation(const ArrayList<double>& m, ModelNode& result)
	{
		if (m.empty()) return;

		for (uint i = 0;i < 4;++i)
		{
			const uint offset = i * 4;
			result.m_Transformation[i] = Vector4(
				(float)m[offset],
				(float)m[offset + 1],
				(float)m[offset + 2],
				(float)m[offset + 3]);
		}
	}

	void ModelLoader::loadMesh(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		if (node.mesh < 0) return;

		gltf::Model& model = *info.model;

		if (model.meshes.empty()) return;

		gltf::Mesh& mesh = model.meshes[node.mesh];
		gltf::Primitive primitive = mesh.primitives[0]; // Only supports 1 primitive per mesh for now
		gltf::Accessor& indexAccessor = model.accessors[primitive.indices];
		
		VertexArray* vertexArray = new VertexArray();
		ArrayList<Vertex> vertices;
		ArrayList<uint> indices;

		if (m_DebugMode) std::cout << "\tLoading mesh " << mesh.name << std::endl;

		setupBuffers(info, mesh, vertexArray, vertices, indices);

		const uint meshIndex = result.m_Model.m_Meshes.size();

		Mesh* outMesh = new Mesh(vertexArray);
		outMesh->m_DrawMode = primitive.mode;
		outMesh->m_IndexCount = indexAccessor.count;
		outMesh->m_IndexType = indexAccessor.componentType;
		outMesh->m_IndexBufferOffset = indexAccessor.byteOffset;
		outMesh->m_Vertices = std::move(vertices);
		outMesh->m_Indices = std::move(indices);

		result.m_Model.m_Meshes.push_back(outMesh);
		result.m_Mesh = meshIndex;
	}

	template<typename T>
	static void setIndices(ArrayList<uint>& indices, const unsigned char* rawData, size_t byteLength)
	{
		const T* indicesData = (const T*)(rawData);
		uint indexCount = byteLength / sizeof(T);
		for (size_t i = 0; i < indexCount; ++i)
			indices.push_back((uint)indicesData[i]);
	}

	void ModelLoader::setupBuffers(ModelInfo& info, const gltf::Mesh& mesh, VertexArray* vertexArray,
		ArrayList<Vertex>& vertices, ArrayList<uint>& indices)
	{
		const gltf::Model& model = *info.model;
		const gltf::Primitive& primitive = mesh.primitives[0];
		Map<int, BufferView>& buffers = *info.buffers;

		for (auto [index, bufferView] : buffers)
		{
			if (bufferView.target == GL_ELEMENT_ARRAY_BUFFER)
			{
				bufferView.buffer->bind(GL_ELEMENT_ARRAY_BUFFER);
				vertexArray->setIndexBuffer(bufferView.buffer);

				const gltf::Accessor& indexAccessor = model.accessors[primitive.indices];
				const gltf::BufferView& view = model.bufferViews[indexAccessor.bufferView];
				const gltf::Buffer& buffer = model.buffers[view.buffer];

				const unsigned char* rawData = buffer.data.data() + view.byteOffset;

				switch (indexAccessor.componentType)
				{
					case GL_SHORT:
						setIndices<short>(indices, rawData, view.byteLength);
						break;
					case GL_UNSIGNED_SHORT:
						setIndices<unsigned short>(indices, rawData, view.byteLength);
						break;
					case GL_INT:
						setIndices<int>(indices, rawData, view.byteLength);
						break;
					case GL_UNSIGNED_INT:
						setIndices<unsigned int>(indices, rawData, view.byteLength);
						break;
					default:
						throw UTAD_EXCEPTION("Unsupported index type");
				}
			}
		}

		for (const auto [attribName, accessorIndex] : primitive.attributes)
		{
			const gltf::Accessor& accessor = model.accessors[accessorIndex];
			const gltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
			const uint stride = accessor.ByteStride(bufferView);

			int count = 1;
			if (accessor.type != TINYGLTF_TYPE_SCALAR) count = accessor.type;

			VertexBuffer* buffer = buffers[accessor.bufferView].buffer;
			vertexArray->addVertexBuffer(accessor.bufferView, buffer, stride);

			VertexAttrib attrib;
			if (!getVertexAttrib(attribName, attrib))
			{
				std::cout << "Unknown attribute: " << attribName << std::endl;
				continue;
			}

			const VertexAttribDescription& desc = VertexAttribDescription::of(attrib);

			if (vertices.empty()) vertices.resize(bufferView.byteLength / sizeof(float));
			
			byte* data = (byte*)model.buffers[bufferView.buffer].data.data() + bufferView.byteOffset;
			for (Vertex& vertex : vertices)
			{
				VertexReader::setVertexAttribute(vertex, data, attrib);
				data += stride; // TODO: check
			}

			vertexArray->setVertexAttribute(accessor.bufferView, attrib, accessor.byteOffset);
		}

		if (indices.empty())
		{
			// No indices, so put them manually
			indices.reserve(vertices.size());
			for (int i = 0; i < vertices.size(); ++i)
				indices.push_back(vertices.size());
		}
	}

	bool ModelLoader::getVertexAttrib(const String& attribName, VertexAttrib& attribute) const
	{
		if (m_VertexAttribsByName.find(attribName) == m_VertexAttribsByName.end()) return false;
		attribute = m_VertexAttribsByName.at(attribName);
		return true;
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

	static Texture2D* toTexture2D(ModelInfo& info, int texIndex, Texture2D* defaultTexture)
	{
		if (texIndex < 0) return defaultTexture;
		auto& textures = *info.textures;
		Texture2D* texture = textures[texIndex];
		return texture != nullptr ? texture : defaultTexture;
	}

	void ModelLoader::loadMaterial(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		if (node.mesh < 0) return;

		gltf::Model& model = *info.model;
		gltf::Mesh& mesh = model.meshes[node.mesh];

		if (model.materials.empty()) return;
		if (mesh.primitives[0].material < 0) return;

		gltf::Material& mat = model.materials[mesh.primitives[0].material];
		gltf::PbrMetallicRoughness& pbr = mat.pbrMetallicRoughness;

		if (m_DebugMode) std::cout << "\tLoading material " << mat.name << std::endl;

		Material* material = Assets::getMaterial(mesh.name);
		if (material == nullptr) material = Assets::createMaterial(mesh.name);

		material->albedo(toColor(pbr.baseColorFactor));
		material->albedoMap(toTexture2D(info, pbr.baseColorTexture.index, material->albedoMap()));
		material->normalScale(static_cast<float>(mat.normalTexture.scale));
		material->normalMap(toTexture2D(info, mat.normalTexture.index, material->normalMap()));
		material->emissiveColor(toColor(mat.emissiveFactor));
		material->emissiveMap(toTexture2D(info, mat.emissiveTexture.index, material->emissiveMap()));
		material->roughness(static_cast<float>(pbr.roughnessFactor));
		material->metallic(static_cast<float>(pbr.metallicFactor));
		material->metallicRoughnessMap(toTexture2D(info, pbr.metallicRoughnessTexture.index, material->metallicRoughnessMap()));
		material->occlusion(static_cast<float>(mat.occlusionTexture.strength));
		material->occlusionMap(toTexture2D(info, mat.occlusionTexture.index, material->occlusionMap()));
		material->alpha(mat.alphaMode == "OPAQUE" ? 1 : static_cast<float>(mat.alphaCutoff));
	
		const uint materialIndex = result.m_Model.m_Materials.size();
		result.m_Model.m_Materials.push_back(material);
		result.m_Mesh = materialIndex;
	}

	void ModelLoader::loadBuffers(ModelInfo& info)
	{
		gltf::Model& model = *info.model;
		Map<int, BufferView>& bufferViews = *info.buffers;
		Map<int, Buffer*> buffers;

		if(m_DebugMode) std::cout << "\tLoading buffers... (" << model.bufferViews.size() << ")" << std::endl;

		for (size_t i = 0; i < model.bufferViews.size(); ++i)
		{
			const gltf::BufferView& bufferView = model.bufferViews[i];
			if (bufferView.target == 0) continue;

			std::cout << "\t\tLoading buffer: '" << bufferView.name << "' (" << targetToStr(bufferView.target) << ")" << std::endl;

			Buffer* buffer = createGLBuffer(bufferView, model.buffers[bufferView.buffer]);

			BufferView view = {};
			view.target = bufferView.target;
			view.offset = bufferView.byteOffset;
			view.size = bufferView.byteLength;
			view.buffer = buffer;

			bufferViews[i] = std::move(view);
		}
	}

	Buffer* ModelLoader::createGLBuffer(const gltf::BufferView& bufferView, const gltf::Buffer& buffer)
	{
		Buffer* result = new Buffer();

		BufferAllocInfo allocInfo = {};
		allocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;
		allocInfo.size = bufferView.byteLength;
		allocInfo.data = const_cast<unsigned char*>(buffer.data.data() + bufferView.byteOffset);

		result->allocate(allocInfo);

		return result;
	}

	void ModelLoader::loadTextures(ModelInfo& info)
	{
		gltf::Model& model = *info.model;
		auto& textures = *info.textures;

		if (m_DebugMode) std::cout << "Loading textures... (" << model.textures.size() << ")" << std::endl;

		for (int i = 0;i < model.textures.size();++i)
		{
			const gltf::Texture& tex = model.textures[i];

			if (tex.source < 0) continue;
			//if (textures.find(tex.name) != textures.end()) continue;

			if (m_DebugMode) std::cout << "\t\tLoading texture '" << tex.name << "'" << std::endl;

			gltf::Image& img = model.images[tex.source];
			gltf::Sampler& sampler = model.samplers[tex.sampler];

			GLenum pixelFormat = getPixelFormatFrom(img.component);

			Texture2D* texture = new Texture2D();
			texture->wrap(GL_TEXTURE_WRAP_S, sampler.wrapS);
			texture->wrap(GL_TEXTURE_WRAP_T, sampler.wrapT);
			if(sampler.minFilter != -1) texture->filter(GL_TEXTURE_MIN_FILTER, sampler.minFilter);
			if(sampler.magFilter != -1) texture->filter(GL_TEXTURE_MAG_FILTER, sampler.magFilter);

			TextureAllocInfo allocInfo = {};
			allocInfo.format = toSizedFormat(pixelFormat, img.bits, img.pixel_type);
			allocInfo.width = img.width;
			allocInfo.height = img.height;

			texture->allocate(allocInfo);

			Texture2DUpdateInfo updateInfo = {};
			updateInfo.format = pixelFormat;
			updateInfo.level = 0;
			updateInfo.type = img.pixel_type;
			updateInfo.pixels = img.image.data();

			texture->update(updateInfo);
			texture->generateMipmaps();

			textures.push_back(texture);
		}
	}
}