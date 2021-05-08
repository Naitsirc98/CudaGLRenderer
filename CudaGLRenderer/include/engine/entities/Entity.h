#pragma once

#include "Transform.h"
#include "Script.h"

namespace utad
{
	using EntityID = uint;

	const EntityID ENTITY_INVALID_ID = 0;

	class Entity
	{
	private:
		EntityID m_ID{ENTITY_INVALID_ID};
		uint m_ChildIndex{UINT32_MAX};
		String m_Name;
		Entity* m_Parent{nullptr};
		ArrayList<Entity*> m_Children;
		Transform m_Transform;
		ArrayList<Script*> m_Scripts;
	public:
		Entity(const String& name = "");
		Entity(const Entity& other) = delete;
		Entity& operator=(const Entity& other) = delete;
		~Entity();
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
	private:
		void update();
	};
}