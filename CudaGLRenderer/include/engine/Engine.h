#pragma once

#include "Common.h"
#include "graphics/Window.h"

namespace utad
{
	class Engine
	{
	public:
		static void launch();
	private:
		Window* m_Window{nullptr};
		float m_UpdateDelay{0};
		int m_UPS{ 0 };
		volatile bool m_Active{ false };
	private:
		Engine();
		void start();
		void run();
		void update(float deltaTime);
		void render();
		void shutdown();
	};

}