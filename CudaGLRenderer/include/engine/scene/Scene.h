#pragma once
#include "engine/entities/EntityPool.h"

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
	private:
		Scene();
		~Scene();
	public:
		Entity* createEntity(const String& name = "");
		void destroyEntity(Entity* entity);
		void destroyAllEntities();
		Entity* find(const String& name) const;
	private:
		void update();
		void render();
	};
}