#pragma once

#include "Common.h"
#include "engine/Time.h"
#include "Application.h"
#include "graphics/Window.h"
#include "events/EventSystem.h"
#include "events/Input.h"
#include "assets/AssetsManager.h"
#include "scene/Scene.h"
#include "engine/assets/Primitives.h"

#define UTAD_EXIT_SUCCESS 0
#define UTAD_EXIT_FAILURE 1
#define UTAD_NOT_EXECUTED 2

namespace utad
{
	class Engine
	{
	public:
		static int launch();
		static int launch(Application& application);
	private:
		Application& m_App;
		Window* m_Window{nullptr};
		EventSystem* m_EventSystem{nullptr};
		Input* m_Input{nullptr};
		Scene* m_Scene{nullptr};
		float m_UpdateDelay{0};
		uint m_UPS{0};
		uint m_FPS{0};
		volatile bool m_Active{ false };
		bool m_AppExited{ false };
		float m_DebugTimer;
	private:
		Engine(Application& application);
		~Engine();
		void start();
		void run();
		void update(float deltaTime);
		void render();
		void shutdown();
		void showDebugInfo();
	};

}