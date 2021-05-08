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
		String m_Name;
		Entity* m_Parent{nullptr};
		ArrayList<Entity*> m_Children;
		Transform m_Transform;
		ArrayList<Script*> m_Scripts;
	public:
		Entity(const String& name = "");
		Entity(String&& name = "");
		~Entity();
		EntityID id() const;
		const String& name() const;
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
	};
}