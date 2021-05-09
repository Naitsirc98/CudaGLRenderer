#pragma once

#include "Mesh.h"
#include "engine/graphics/Texture.h"

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
	private:
		AssetsManager();
		~AssetsManager();
	public:
		static Model* createModel(const String& name, const String& path);
		static Model* getModel(const String& name);
	};

	using Assets = AssetsManager;
}