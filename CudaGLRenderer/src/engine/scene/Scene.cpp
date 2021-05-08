#include "engine/scene/Scene.h"

namespace utad
{
	Scene* Scene::s_Instance = nullptr;

	Scene* Scene::init()
	{
		s_Instance = new Scene();
		return s_Instance;
	}

	void Scene::destroy()
	{
		UTAD_DELETE(s_Instance);
	}

	Scene& Scene::get()
	{
		return *s_Instance;
	}

	Scene::Scene()
	{
		m_Entities.reserve(16);
	}

	Scene::~Scene()
	{
		for (Entity* entity : m_Entities)
		{
			UTAD_DELETE(entity);
		}
		m_Entities.clear();
		m_EntitiesByName.clear();
	}

	const ArrayList<Entity*>& Scene::entities() const
	{
		return m_Entities;
	}

	void Scene::addEntity(Entity* entity)
	{
		if (entity->m_SceneIndex != UINT32_MAX) return;
		entity->m_SceneIndex = m_Entities.size();
		m_Entities.push_back(entity);
		if (!entity->name().empty())
		{
			if (find(entity->name()) != nullptr) throw UTAD_EXCEPTION("Entity names must be unique");
			m_EntitiesByName[entity->name()] = entity;
		}
	}

	void Scene::removeEntity(Entity* entity)
	{
		if (entity->m_SceneIndex == UINT32_MAX) return;
		m_Entities.erase(m_Entities.begin() + entity->m_SceneIndex);
		entity->m_SceneIndex = UINT32_MAX;
		if (!entity->name().empty())
		{
			m_EntitiesByName[entity->name()] = nullptr;
		}
	}

	void Scene::removeAllEntities()
	{
		for (Entity* entity : m_Entities)
		{
			entity->m_SceneIndex = UINT32_MAX;
		}
		m_Entities.clear();
		m_EntitiesByName.clear();
	}

	bool Scene::contains(Entity* entity) const
	{
		return entity->m_SceneIndex != UINT32_MAX && m_Entities[entity->m_SceneIndex] == entity;
	}

	Entity* Scene::find(const String& name) const
	{
		if (m_EntitiesByName.find(name) == m_EntitiesByName.end()) return nullptr;
		return m_EntitiesByName.at(name);
	}

	void Scene::update()
	{
		for (Entity* entity : m_Entities)
		{
			entity->update();
		}
	}

	void Scene::render()
	{
		for (Entity* entity : m_Entities)
		{
			entity->render();
		}
	}

}