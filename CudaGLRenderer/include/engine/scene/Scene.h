#pragma once

#include "engine/entities/EntityPool.h"
#include "Camera.h"
#include "Light.h"
#include "engine/assets/Skybox.h"
#include "engine/graphics/MeshRenderer.h"
#include "engine/graphics/SkyboxRenderer.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/raytracing/RayTracer.h"

namespace utad
{
	struct SceneSetup
	{
		friend class Scene;

		Camera camera;
		Light dirLight;
		bool enableDirLight;
		ArrayList<Light> pointLights;
		Skybox* skybox{nullptr};
		ArrayList<PostFX> postEffects;
		SortedMap<String, RenderQueue*> renderQueues;

	private:
		SceneSetup() 
		{
			RenderQueue* defaultRenderQueue = new RenderQueue();
			defaultRenderQueue->name = DEFAULT_RENDER_QUEUE;
			defaultRenderQueue->enabled = true;
			renderQueues[DEFAULT_RENDER_QUEUE] = defaultRenderQueue;
		}

		~SceneSetup()
		{
			for (auto [name, queue] : renderQueues)
			{
				UTAD_DELETE(queue);
			}
			renderQueues.clear();
		}
	};

	class Scene
	{
		friend class Engine;
	private:
		static Scene* s_Instance;
	private:
		static Scene* init();
		static void destroy();
	public:
		static Scene& get();
	private:
		EntityPool* m_EntityPool;
		MeshRenderer* m_MeshRenderer;
		SkyboxRenderer* m_SkyboxRenderer;
		PostFXRenderer* m_PostFXRenderer;
		RayTracer* m_RayTracer;
		SceneSetup m_RenderInfo;
	private:
		Scene();
		~Scene();
	public:
		Entity* createEntity(const String& name = "");
		void destroyEntity(Entity* entity);
		void destroyAllEntities();
		Entity* find(const String& name) const;
		Camera& camera();
		Light& dirLight();
		void enableDirLight(bool enable);
		ArrayList<Light>& pointLights();
		Skybox* skybox() const;
		void setSkybox(Skybox* skybox, bool deleteExisting = true);
		ArrayList<PostFX>& postEffects();
		RenderQueue& getRenderQueue(const String& name);
	private:
		void update();
		void lastUpdate();
		void render();
		void updateOctree();
	};
}