#pragma once
#include "engine/entities/EntityPool.h"
#include "engine/graphics/MeshRenderer.h"
#include "engine/graphics/SkyboxRenderer.h"
#include "Camera.h"
#include "Light.h"
#include "Skybox.h"

namespace utad
{
	struct RenderInfo
	{
		friend class Scene;

		Camera camera;
		Light dirLight;
		bool enableDirLight;
		ArrayList<Light> pointLights;
		Skybox skybox[1];

	private:
		RenderInfo() {}
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
		RenderInfo m_RenderInfo;
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
	private:
		void update();
		void lastUpdate();
		void render();
	};
}