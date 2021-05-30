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
		m_MeshRenderer = new MeshRenderer();
		m_SkyboxRenderer = new SkyboxRenderer();
	}

	Scene::~Scene()
	{
		UTAD_DELETE(m_EntityPool);
		UTAD_DELETE(m_MeshRenderer);
		UTAD_DELETE(m_SkyboxRenderer);
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

	Camera& Scene::camera()
	{
		return m_Camera;
	}

	void Scene::update()
	{
		for (Entity* entity : m_EntityPool->entities())
		{
			if (entity->id() == NULL) continue;
			entity->update();
		}
	}

	void Scene::lastUpdate()
	{
		m_Camera.update();

		for (Entity* entity : m_EntityPool->entities())
		{
			if (entity->id() == NULL) continue;
			entity->meshView().update(*m_MeshRenderer);
		}
	}

	void Scene::render()
	{
		m_MeshRenderer->render(m_Camera);
		m_SkyboxRenderer->render(m_Camera);
	}

}