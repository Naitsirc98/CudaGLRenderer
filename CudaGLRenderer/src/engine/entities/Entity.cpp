#include "engine/entities/Entity.h"
#include <mutex>

namespace utad
{
	static std::mutex g_Mutex;

	static EntityID nextEntityID()
	{
		static volatile EntityID nextID = 1;

		std::unique_lock lock(g_Mutex);
		EntityID id = nextID++;
		return id;
	}

	Entity::Entity(const String& name) : m_ID(nextEntityID()), m_Name(name)
	{
		m_Transform.m_Entity = this;
	}

	Entity::~Entity()
	{
		for (Entity* child : m_Children)
		{
			UTAD_DELETE(child);
		}
		m_Children.clear();

		for (Script* script : m_Scripts)
		{
			UTAD_DELETE(script);
		}
		m_Scripts.clear();

		if (m_Parent != nullptr)
		{
			m_Parent->removeChild(this);
			m_Parent = nullptr;
		}

		m_ID = ENTITY_INVALID_ID;
	}

	EntityID Entity::id() const
	{
		return m_ID;
	}

	const String& Entity::name() const
	{
		return m_Name;
	}

	Entity* Entity::parent() const
	{
		return m_Parent;
	}

	bool Entity::hasParent() const
	{
		return m_Parent != nullptr;
	}

	const Transform& Entity::transform() const
	{
		return m_Transform;
	}

	Transform& Entity::transform()
	{
		return m_Transform;
	}

	const ArrayList<Entity*> Entity::children() const
	{
		return m_Children;
	}

	void Entity::addChild(Entity* child)
	{
		if (child->m_Parent == this) return;
		if (child->m_Parent != nullptr) child->m_Parent->removeChild(child);
		child->m_ChildIndex = m_Children.size();
		m_Children.push_back(child);
		child->m_Parent = this;
	}

	void Entity::removeChild(Entity* child)
	{
		if (child->m_Parent != this) return;
		m_Children.erase(m_Children.begin() + child->m_ChildIndex);
		child->m_Parent = nullptr;
		m_ChildIndex = UINT32_MAX;
	}

	void Entity::removeAllChildren(Entity* child)
	{
		for (Entity* child : m_Children)
		{
			child->m_Parent = nullptr;
			child->m_ChildIndex = UINT32_MAX;
		}
		m_Children.clear();
	}

	const ArrayList<Script*> Entity::scripts() const
	{
		return m_Scripts;
	}

	void Entity::addScript(Script* script)
	{
		if (script->m_Entity == this) return;
		if (script->m_Entity != nullptr) script->m_Entity->removeScript(script);
		script->m_Entity = this;
		script->m_Index = m_Scripts.size();
		m_Scripts.push_back(script);
	}

	void Entity::removeScript(Script* script)
	{
		if (script->m_Entity != this) return;
		m_Scripts.erase(m_Scripts.begin() + script->m_Index);
		script->m_Entity = nullptr;
		script->m_Index = UINT32_MAX;
	}

	void Entity::update()
	{
		m_Transform.computeModelMatrix();
		
		for (Script* script : m_Scripts)
		{
			script->onUpdate();
		}

		for (Entity* child : m_Children)
		{
			child->update();
		}
	}

	void Entity::render()
	{
		// TODO
	}
}