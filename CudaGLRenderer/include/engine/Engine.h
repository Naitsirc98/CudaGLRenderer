#pragma once

#include "Common.h"
#include "graphics/Window.h"

#define UTAD_EXIT_SUCCESS 0
#define UTAD_EXIT_FAILURE 1
#define UTAD_NOT_EXECUTED 2

namespace utad
{
	class Engine
	{
	public:
		static int launch();
	private:
		Window* m_Window{nullptr};
		float m_UpdateDelay{0};
		uint m_UPS{0};
		uint m_FPS{0};
		volatile bool m_Active{ false };
		float m_DebugTimer;
	private:
		Engine();
		void start();
		void run();
		void update(float deltaTime);
		void render();
		void shutdown();
		void showDebugInfo();
	};

}