#pragma once
#include "engine/entities/EntityPool.h"
#include "Camera.h"
#include "engine/graphics/MeshRenderer.h"
#include "engine/graphics/SkyboxRenderer.h"

namespace utad
{
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
		Camera m_Camera;
	private:
		Scene();
		~Scene();
	public:
		Entity* createEntity(const String& name = "");
		void destroyEntity(Entity* entity);
		void destroyAllEntities();
		Entity* find(const String& name) const;
		Camera& camera();
	private:
		void update();
		void lastUpdate();
		void render();
	};
}