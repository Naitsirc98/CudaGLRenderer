#pragma once

#include "Transform.h"
#include "Script.h"
#include "MeshView.h"

namespace utad
{
	using EntityID = uint;

	const EntityID ENTITY_INVALID_ID = 0;

	class Entity
	{
		friend class EntityPool;
		friend class Scene;
	public:
		static Entity* create(const String& name = "");
	private:
		EntityID m_ID{ENTITY_INVALID_ID};
		uint m_SceneIndex{UINT32_MAX};
		uint m_ChildIndex{UINT32_MAX};
		String m_Name;
		Entity* m_Parent{nullptr};
		ArrayList<Entity*> m_Children;
		Transform m_Transform;
		ArrayList<Script*> m_Scripts;
		MeshView m_MeshView;
		bool m_Enabled{true};
	private:
		Entity();
		Entity(const Entity& other) = delete;
		Entity& operator=(const Entity& other) = delete;
		~Entity();
	public:
		EntityID id() const;
		const String& name() const;
		bool hasParent() const;
		Entity* parent() const;
		const ArrayList<Entity*> children() const;
		void addChild(Entity* child);
		void removeChild(Entity* child);
		void removeAllChildren(Entity* child);
		const Transform& transform() const;
		Transform& transform();
		const ArrayList<Script*> scripts() const;
		void addScript(Script* script);
		void removeScript(Script* script);
		MeshView& meshView();
		bool enabled() const;
		void setEnabled(bool enable);
		void destroy();
	private:
		void init(EntityID id, const String& name, uint sceneIndex);
		void update();
		void onDestroy();
	};
}