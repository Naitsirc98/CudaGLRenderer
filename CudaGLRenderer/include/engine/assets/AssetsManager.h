#pragma once

#include "Model.h"
#include "engine/graphics/Texture.h"
#include "engine/assets/ModelLoader.h"
#include "SkyboxLoader.h"

namespace utad
{
	class AssetsManager
	{
		friend class Engine;
	private:
		static AssetsManager* s_Instance;
	public:
		static AssetsManager* init();
		static void destroy();
		static AssetsManager& get();
	private:
		Map<String, Model*> m_Models;
		Map<String, Model*> m_ModelsByName;
		Map<String, Material*> m_Materials;
		Texture2D* m_WhiteTexture;
		Texture2D* m_BlackTexture;
		SkyboxLoader* m_SkyboxLoader;
	private:
		AssetsManager();
		~AssetsManager();
	public:
		static Model* createModel(const String& name, const String& path);
		static Model* getModel(const String& name);
		static Material* createMaterial(const String& name);
		static Material* getMaterial(const String& name);
		static Texture2D* getWhiteTexture();
		static Texture2D* getBlackTexture();
		static Skybox* loadSkybox(const String& path, const SkyboxLoadInfo& loadInfo = DEFAULT_SKYBOX_LOAD_INFO);
	private:
		static void initMaterialTextures(Material* material);
	};

	using Assets = AssetsManager;
}