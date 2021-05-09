#pragma once

#include "engine/entities/EntityPool.h"
#include "engine/Common.h"

namespace utad
{
	EntityPool::EntityPool()
	{
		m_Entities.reserve(16);
	}

	EntityPool::~EntityPool()
	{
		destroyAll();
	}

	Entity* EntityPool::create(const String& name)
	{
		if (find(name) != nullptr) throw UTAD_EXCEPTION(String("Entity named ").append(name).append(" already exists"));

		Entity* entity;
		uint index;

		if (m_FreeIndices.empty())
		{
			entity = new Entity();
			index = m_Entities.size();
			m_Entities.push_back(entity);
		}
		else
		{
			index = m_FreeIndices.front();
			entity = m_Entities[index];
			m_FreeIndices.pop_front();
		}

		entity->init(m_NextID++, name, index);

		if (!entity->name().empty())
		{
			if (find(entity->name()) != nullptr) throw UTAD_EXCEPTION("Entity names must be unique");
			m_EntitiesByName[entity->name()] = entity;
		}

		return entity;
	}

	void EntityPool::destroy(Entity* entity)
	{
		if (entity->m_ID == NULL) return;
		if (!entity->name().empty()) m_EntitiesByName[entity->name()] = nullptr;
		entity->onDestroy();
	}

	void EntityPool::destroyAll()
	{
		for (Entity* entity : m_Entities)
		{
			entity->onDestroy();
			UTAD_DELETE(entity);
		}
		m_Entities.clear();
		m_EntitiesByName.clear();
		m_FreeIndices.clear();
	}

	Entity* EntityPool::find(const String& name) const
	{
		if (m_EntitiesByName.find(name) == m_EntitiesByName.end()) return nullptr;
		return m_EntitiesByName.at(name);
	}

	const ArrayList<Entity*>& EntityPool::entities() const
	{
		return m_Entities;
	}

}
