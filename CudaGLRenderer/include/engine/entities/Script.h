#pragma once
#include "engine/Common.h"

namespace utad
{
	class Script
	{
		friend class Entity;
	private:
		Entity* m_Entity;
		uint m_Index{UINT32_MAX};
	public:
		virtual ~Script() { onDestroy(); }
		Entity* entity() const { return m_Entity; }
	protected:
		virtual void onStart() {}
		virtual void onUpdate() = 0;
		virtual void onDestroy() {}
	};

	class LambdaScript : public Script
	{
	private:
		Function<void> m_OnUpdate;
		Function<void> m_OnDestroy;
	public:
		LambdaScript(const Function <void>& onUpdateFunc = LambdaScript::defaultFunction) : m_OnUpdate(onUpdateFunc) { }
		void onUpdate() override { m_OnUpdate(); }
		LambdaScript& setOnUpdate(const Function<void>& function) { this->m_OnUpdate = function; return *this; }
		LambdaScript& setOnDestroy(const Function<void>& function) { this->m_OnDestroy = function; return *this;}
	private:
		static void defaultFunction() {}
	};
}
