#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"

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
		return m_RenderInfo.camera;
	}

	Light& Scene::dirLight()
	{
		return m_RenderInfo.dirLight;
	}

	void Scene::enableDirLight(bool enable)
	{
		m_RenderInfo.enableDirLight = enable;
	}

	ArrayList<Light>& Scene::pointLights()
	{
		return m_RenderInfo.pointLights;
	}

	Skybox* Scene::skybox() const
	{
		return m_RenderInfo.skybox;
	}

	void Scene::setSkybox(Skybox* skybox, bool deleteExisting)
	{
		if (deleteExisting)
		{
			UTAD_DELETE(m_RenderInfo.skybox);
		}
		m_RenderInfo.skybox = skybox;
	}

	void Scene::update()
	{
		for (Entity* entity : m_EntityPool->entities())
		{
			if (entity->id() == NULL) continue;
			if (!entity->enabled()) continue;
			entity->update();
		}
	}

	void Scene::lastUpdate()
	{
		m_RenderInfo.camera.update();

		m_MeshRenderer->clearRenderQueues();

		for (Entity* entity : m_EntityPool->entities())
		{
			if (entity->id() == NULL) continue;
			if (!entity->enabled()) continue;
			entity->meshView().update(*m_MeshRenderer);
		}
	}

	void Scene::render()
	{
		const Window& window = Window::get();
		Framebuffer::bindDefault();
		glViewport(0, 0, window.width(), window.height());
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		m_MeshRenderer->render(m_RenderInfo);
		m_SkyboxRenderer->render(m_RenderInfo);
	}

}