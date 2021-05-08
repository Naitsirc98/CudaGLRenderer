#pragma once
#include "engine/Common.h"
#include "engine/entities/Entity.h"

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
		ArrayList<Entity*> m_Entities;
		Map<String, Entity*> m_EntitiesByName;
	private:
		Scene();
		~Scene();
	public:
		const ArrayList<Entity*>& entities() const;
		void addEntity(Entity* entity);
		void removeEntity(Entity* entity);
		void removeAllEntities();
		bool contains(Entity* entity) const;
		Entity* find(const String& name) const;
	private:
		void update();
		void render();
	};
}