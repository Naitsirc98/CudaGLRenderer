#include "engine/assets/AssetsManager.h"
#include "engine/assets/ModelLoader.h"
#include "engine/assets/Image.h"
#include "engine/assets/MeshPrimitives.h"

namespace utad
{
	AssetsManager* AssetsManager::s_Instance;

	static Texture2D* createTextureFromImage(Image* image)
	{
		Texture2D* texture = new Texture2D();
		
		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_SRGB8_ALPHA8;
		allocInfo.width = image->width();
		allocInfo.height = image->height();
		allocInfo.levels = 1;

		texture->allocate(allocInfo);

		Texture2DUpdateInfo updateInfo = {};
		updateInfo.format = GL_RGBA;
		updateInfo.type = GL_UNSIGNED_BYTE;
		updateInfo.level = 0;
		updateInfo.pixels = image->pixels();

		texture->update(updateInfo);

		texture->wrap(GL_REPEAT);
		texture->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		texture->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		UTAD_DELETE(image);

		return texture;
	}

	AssetsManager* AssetsManager::init()
	{
		MeshPrimitives::initMeshes();
		s_Instance = new AssetsManager();
		s_Instance->m_WhiteTexture = createTextureFromImage(ImageFactory::createWhiteImage(GL_RGBA));
		s_Instance->m_BlackTexture = createTextureFromImage(ImageFactory::createBlackImage(GL_RGBA));
		s_Instance->m_SkyboxLoader = new SkyboxLoader();
		s_Instance->m_SkyboxLoader->init();
		return s_Instance;
	}

	void AssetsManager::destroy()
	{
		UTAD_DELETE(s_Instance);
		MeshPrimitives::destroyMeshes();
	}

	AssetsManager& AssetsManager::get()
	{
		return *s_Instance;
	}

	AssetsManager::AssetsManager()
	{

	}

	AssetsManager::~AssetsManager()
	{
		for (auto[name, model] : m_Models)
		{
			UTAD_DELETE(model);
		}
		m_Models.clear();
		m_ModelsByName.clear();

		for (auto [name, material] : m_Materials)
		{
			UTAD_DELETE(material);
		}
		m_Materials.clear();

		UTAD_DELETE(m_WhiteTexture);
		UTAD_DELETE(m_BlackTexture);

		UTAD_DELETE(m_SkyboxLoader);
	}

	Model* AssetsManager::createModel(const String& name, const String& path)
	{
		Model* model = s_Instance->m_Models[path];
		if (model != nullptr) return model;
		model = getModel(name);
		if (model != nullptr) return model;
		
		model = ModelLoader().load(name, path);
		s_Instance->m_Models[path] = model;
		s_Instance->m_ModelsByName[name] = model;
		
		return model;
	}

	Model* AssetsManager::getModel(const String& name)
	{
		auto& models = s_Instance->m_Models;
		if (models.find(name) == models.end()) return nullptr;
		return models[name];
	}

	static MaterialID g_MaterialID = 1;

	Material* AssetsManager::createMaterial(const String& name)
	{
		Material* material = getMaterial(name);
		if (material != nullptr) throw UTAD_EXCEPTION(String("Material ").append(name).append(" already exists!"));
		auto& materials = s_Instance->m_Materials;
		material = new Material(g_MaterialID++);
		initMaterialTextures(material);
		materials[name] = material;
		return material;
	}

	Material* AssetsManager::getMaterial(const String& name)
	{
		auto& materials = s_Instance->m_Materials;
		if (materials.find(name) == materials.end()) return nullptr;
		return materials.at(name);
	}

	Texture2D* AssetsManager::getWhiteTexture()
	{
		return s_Instance->m_WhiteTexture;
	}

	Texture2D* AssetsManager::getBlackTexture()
	{
		return s_Instance->m_BlackTexture;
	}

	Skybox* AssetsManager::loadSkybox(const String& path, const SkyboxLoadInfo& loadInfo)
	{
		return s_Instance->m_SkyboxLoader->loadSkybox(path, loadInfo);
	}

	void AssetsManager::initMaterialTextures(Material* material)
	{
		material->albedoMap(s_Instance->m_WhiteTexture);
		material->metallicRoughnessMap(s_Instance->m_WhiteTexture);
		material->occlusionMap(s_Instance->m_WhiteTexture);
		material->normalMap(s_Instance->m_WhiteTexture);
		material->emissiveMap(s_Instance->m_BlackTexture);
	}
}