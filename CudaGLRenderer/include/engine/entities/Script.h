#pragma once

namespace utad
{
	class Script
	{
		friend class Entity;
	private:
		Entity* m_Entity;
	public:
		Entity* entity() const { return m_Entity; }
		virtual void onUpdate() = 0;
	};

}
