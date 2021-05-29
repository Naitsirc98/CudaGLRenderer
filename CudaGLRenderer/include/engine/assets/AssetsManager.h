#pragma once

#include "Model.h"
#include "engine/graphics/Texture.h"
#include "engine/assets/ModelLoader.h"

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
	private:
		AssetsManager();
		~AssetsManager();
	public:
		static Model* createModel(const String& name, const String& path);
		static Model* getModel(const String& name);
		static Material* createMaterial(const String& name);
		static Material* getMaterial(const String& name);
	};

	using Assets = AssetsManager;
}