#pragma once

namespace utad
{
	class Application
	{
		friend class Engine;
	protected:
		virtual void onStart() {}
		virtual void onUpdate() {}
		virtual void onRender() {}
		virtual void onExit() {}
	};
}