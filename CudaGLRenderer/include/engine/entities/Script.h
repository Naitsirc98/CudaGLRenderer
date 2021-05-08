#pragma once
#include <functional>

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
		virtual void onUpdate() = 0;
		virtual void onDestroy() {}
	};

	template<typename R, typename ...Args>
	using Function = std::function<R(Args...)>;

	class LambdaScript : public Script
	{
	private:
		Function<void> m_OnUpdate;
		Function<void> m_OnDestroy;
	public:
		LambdaScript(const Function <void>& onUpdateFunc = LambdaScript::defaultFunction) : m_OnUpdate(onUpdateFunc) { }
		void onUpdate() override { m_OnUpdate(); }
		LambdaScript& setOnUpdate(const Function<void>& function) { this->m_OnUpdate = function; }
		LambdaScript& setOnDestroy(const Function<void>& function) { this->m_OnDestroy = function; }
	private:
		static void defaultFunction() {}
	};
}
