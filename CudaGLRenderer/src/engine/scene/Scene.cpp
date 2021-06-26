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
		m_PostFXRenderer = new PostFXRenderer();
		m_RayTracer = new RayTracer();
	}

	Scene::~Scene()
	{
		UTAD_DELETE(m_EntityPool);
		UTAD_DELETE(m_MeshRenderer);
		UTAD_DELETE(m_SkyboxRenderer);
		UTAD_DELETE(m_PostFXRenderer);
		UTAD_DELETE(m_RayTracer);
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

	ArrayList<PostFX>& Scene::postEffects()
	{
		return m_RenderInfo.postEffects;
	}

	RenderQueue& Scene::getRenderQueue(const String& name)
	{
		if (m_RenderInfo.renderQueues.find(name) == m_RenderInfo.renderQueues.end())
		{
			RenderQueue* queue = new RenderQueue();
			queue->name = name;
			m_RenderInfo.renderQueues[name] = queue;
		}

		return *m_RenderInfo.renderQueues[name];
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

		for (auto [name, queue] : m_RenderInfo.renderQueues)
		{
			queue->commands.clear();
		}

		for (Entity* entity : m_EntityPool->entities())
		{
			if (entity->id() == NULL) continue;
			if (!entity->enabled()) continue;
			entity->meshView().prepareForRender(*this);
		}

		if (Graphics::getRenderMethod() == RenderMethod::RayTracing)
		{
			m_RayTracer->prepareColorBuffer();
			//updateOctree();
		}
	}

	void Scene::render()
	{
		if (Graphics::getRenderMethod() == RenderMethod::Rasterization)
		{
			m_MeshRenderer->render(m_RenderInfo);
			m_SkyboxRenderer->render(m_RenderInfo);
			m_PostFXRenderer->render(m_RenderInfo);
		}
		else
		{
			m_RayTracer->render(m_RenderInfo);
		}
	}

	void Scene::updateOctree()
	{
		RenderQueue* queue = m_RenderInfo.renderQueues[DEFAULT_RENDER_QUEUE];
		const Matrix4& view = m_RenderInfo.camera.viewMatrix();

		for (RenderCommand& command : queue->commands)
		{
			command.aabb->update(view * *command.transformation);
		}
	}

}