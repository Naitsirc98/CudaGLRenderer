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
		m_EntityPool = new EntityPool();
	}

	Scene::~Scene()
	{
	}

	Entity* Scene::createEntity(const String& name)
	{
		return m_EntityPool->create(name);
	}

	void Scene::destroyEntity(Entity* entity)
	{
		m_EntityPool->destroy(entity);
	}

	void Scene::destroyAllEntities()
	{
		m_EntityPool->destroyAll();
	}

	Entity* Scene::find(const String& name) const
	{
		return m_EntityPool->find(name);
	}

	void Scene::update()
	{
		for (Entity* entity : m_EntityPool->entities())
		{
			if(entity->id() != NULL) entity->update();
		}
	}

	void Scene::render()
	{

	}

}