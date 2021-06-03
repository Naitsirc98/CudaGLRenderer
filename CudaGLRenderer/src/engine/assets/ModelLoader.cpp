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
		Map<String, Texture2D*> textures;

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
		Map<int, BufferView>& buffers = *info.buffers;
		gltf::Mesh& mesh = model.meshes[node.mesh];
		gltf::Primitive primitive = mesh.primitives[0]; // Only supports 1 primitive per mesh for now
		gltf::Accessor& indexAccessor = model.accessors[primitive.indices];
		
		VertexArray* vertexArray = new VertexArray();
		vertexArray->bind();

		if (m_DebugMode) std::cout << "\tLoading mesh " << mesh.name << std::endl;

		for (auto [index, bufferView] : buffers)
		{
			if (bufferView.target == GL_ELEMENT_ARRAY_BUFFER)
			{
				bufferView.buffer->bind(GL_ELEMENT_ARRAY_BUFFER);
				vertexArray->setIndexBuffer(bufferView.buffer);
			}
		}

		for (const auto [attribName, accessorIndex] : primitive.attributes)
		{
			gltf::Accessor& accessor = model.accessors[accessorIndex];
			gltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
			const uint stride = accessor.ByteStride(bufferView);

			int count = 1;
			if (accessor.type != TINYGLTF_TYPE_SCALAR) count = accessor.type;

			VertexBuffer* buffer = buffers[accessor.bufferView].buffer;
			vertexArray->addVertexBuffer(accessor.bufferView, buffer, stride);
			
			int location = -1;
			if (attribName == "POSITION") location = 0;
			else if (attribName == "NORMAL") location = 1;
			else if (attribName == "TEXCOORD_0") location = 2;
			else if (attribName == "TANGENT") location = 3;

			if (location == -1) continue;

			VertexAttrib attrib;
			attrib.location = location;
			attrib.count = count;
			attrib.type = accessor.componentType;

			if (location >= 0)
				vertexArray->setVertexAttrib(accessor.bufferView, attrib, location, accessor.byteOffset);
			else
				std::cout << "Unknown attribute: " << attribName << std::endl;
		}

		const uint meshIndex = result.m_Model.m_Meshes.size();

		Mesh* outMesh = new Mesh(vertexArray);
		outMesh->m_DrawMode = primitive.mode;
		outMesh->m_IndexCount = indexAccessor.count;
		outMesh->m_IndexType = indexAccessor.componentType;
		outMesh->m_IndexBufferOffset = indexAccessor.byteOffset;

		result.m_Model.m_Meshes.push_back(outMesh);
		result.m_Mesh = meshIndex;

		vertexArray->unbind();
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
		Texture2D* texture = textures[info.model->textures[texIndex].name];
		return texture != nullptr ? texture : defaultTexture;
	}

	void ModelLoader::loadMaterial(ModelInfo& info, gltf::Node& node, ModelNode& result)
	{
		if (node.mesh < 0) return;

		gltf::Model& model = *info.model;
		gltf::Mesh& mesh = model.meshes[node.mesh];

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
		//material->emissiveColor(toColor(mat.emissiveFactor));
		//material->emissiveMap(toTexture2D(info, mat.emissiveTexture.index, material->emissiveMap()));
		material->roughness(static_cast<float>(pbr.roughnessFactor));
		material->metallic(static_cast<float>(pbr.metallicFactor));
		material->metallicRoughnessMap(toTexture2D(info, pbr.metallicRoughnessTexture.index, material->metallicRoughnessMap()));
		material->occlussion(static_cast<float>(mat.occlusionTexture.strength));
		material->occlussionMap(toTexture2D(info, mat.occlusionTexture.index, material->occlussionMap()));
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

	static GLenum getPixelFormatFrom(int component)
	{
		switch (component)
		{
			case 1: return GL_RED;
			case 2: return GL_RG;
			case 3: return GL_RGB;
			case 4: return GL_RGBA;
		}
		throw UTAD_EXCEPTION(String("Unknown format for components: ").append(std::to_string(component)));
	}

	static GLenum toSizedFormatRED(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_R8;
			case 16: return type == GL_FLOAT ? GL_R16F : GL_R16;
			case 32: return type == GL_FLOAT ? GL_R32F : GL_R32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	static GLenum toSizedFormatRG(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RG8;
			case 16: return type == GL_FLOAT ? GL_RG16F : GL_RG16;
			case 32: return type == GL_FLOAT ? GL_RG32F : GL_RG32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	static GLenum toSizedFormatRGB(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RGB8;
			case 16: return type == GL_FLOAT ? GL_RGB16F : GL_RGB16;
			case 32: return type == GL_FLOAT ? GL_RGB32F : GL_RGB32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	static GLenum toSizedFormatRGBA(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RGBA8;
			case 16: return type == GL_FLOAT ? GL_RGBA16F : GL_RGBA16;
			case 32: return type == GL_FLOAT ? GL_RGBA32F : GL_RGBA32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	static GLenum toSizedFormat(GLenum format, int bits, int type)
	{
		switch (format)
		{
			case GL_RED: return toSizedFormatRED(bits, type);
			case GL_RG: return toSizedFormatRG(bits, type);
			case GL_RGB: return toSizedFormatRGB(bits, type);
			case GL_RGBA: return toSizedFormatRGBA(bits, type);
		}
		throw UTAD_EXCEPTION(String("Unknown combination of format, bits and type: ").append(std::to_string(format)).append(" + ")
			.append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	void ModelLoader::loadTextures(ModelInfo& info)
	{
		gltf::Model& model = *info.model;
		auto& textures = *info.textures;

		if (m_DebugMode) std::cout << "Loading textures... (" << model.textures.size() << ")" << std::endl;

		for (gltf::Texture& tex : model.textures)
		{
			if (tex.source < 0) continue;
			if (textures.find(tex.name) != textures.end()) continue;

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

			textures[tex.name] = texture;
		}
	}
}