#include "engine/assets/AssetsManager.h"
#include "engine/assets/ModelLoader.h"

namespace utad
{
	AssetsManager* AssetsManager::s_Instance;

	AssetsManager* AssetsManager::init()
	{
		s_Instance = new AssetsManager();
		return s_Instance;
	}

	void AssetsManager::destroy()
	{
		UTAD_DELETE(s_Instance);
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
		for (auto [name, model] : m_Models)
		{
			UTAD_DELETE(model);
		}
		m_Models.clear();
	}

	Model* AssetsManager::createModel(const String& name, const String& path)
	{
		Model* model = s_Instance->m_Models[path];
		if (model != nullptr) return model;
		model = getModel(name);
		if (model != nullptr) return model;
		
		Model* model = ModelLoader().load(name, path);
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
}