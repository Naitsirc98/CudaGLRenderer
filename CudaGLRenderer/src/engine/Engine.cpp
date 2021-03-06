#include "engine/Engine.h"
#include "engine/Time.h"
#include <iostream>
#include <exception>

namespace utad
{
	static bool g_Running = false;

	static const float TARGET_DELAY = 1.0f / 60.0f;

	int Engine::launch()
	{
		Application app = {};
		return launch(app);
	}

	int Engine::launch(Application& application)
	{
		if (g_Running) return UTAD_NOT_EXECUTED;

		g_Running = true;

		Engine engine(application);

		int exitCode;

		try
		{
			engine.start();
			engine.run();
			exitCode = UTAD_EXIT_SUCCESS;
		}
		catch (...)
		{
			try
			{
				engine.shutdown();
				std::rethrow_exception(std::current_exception());
			}
			catch (const std::exception& exception)
			{
				std::cout << "Fatal exception: " << exception.what() << std::endl;
				exitCode = UTAD_EXIT_FAILURE;
			}
		}

		g_Running = false;
		return exitCode;
	}

	Engine::Engine(Application& application) : m_App(application)
	{
	}

	Engine::~Engine()
	{
		shutdown();
	}

	void Engine::start()
	{
		m_EventSystem = EventSystem::init();
		m_Window = Window::init();
		m_Input = Input::init();
		AssetsManager::init();
		m_Scene = Scene::init();
		m_Window->show();
		Graphics::init();
		UIRenderer::init();
		m_App.onStart();
	}

	void Engine::run()
	{
		const Window* window = m_Window;
		float lastFrame = 0;
		m_DebugTimer = Time::time();

		while (!window->shouldClose())
		{
			const float now = Time::time();
			const float deltaTime = now - lastFrame;
			lastFrame = now;
			Time::s_DeltaTime = deltaTime;

			update(deltaTime);

			render();
			++m_FPS;

			++Time::s_Frame;
		
			showDebugInfo();
		}
	}

	void Engine::update(float deltaTime)
	{
		m_UpdateDelay += deltaTime;
		bool wasUpdated = false;

		while (m_UpdateDelay >= TARGET_DELAY)
		{
			m_EventSystem->update();

			m_Input->update();

			m_Scene->update();

			m_App.onUpdate();

			// UPDATE
			m_UpdateDelay -= TARGET_DELAY;
			++m_UPS;
			wasUpdated = true;
		}

		if (wasUpdated) m_Scene->lastUpdate();
	}

	void Engine::render()
	{
		Graphics::begin();
		{
			m_Scene->render();
			m_App.onRender();
		}
		Graphics::end();

		if (Graphics::getRenderMethod() == RenderMethod::Rasterization)
		{
			UIRenderer::get().render();
			Window::get().swapBuffers();
		}
	}

	void Engine::shutdown()
	{
		if (!m_AppExited)
		{
			m_App.onExit();
			m_AppExited = true;
		}

		Scene::destroy();
		m_Scene = nullptr;

		UIRenderer::destroy();

		AssetsManager::destroy();

		Input::destroy();
		m_Input = nullptr;

		Graphics::destroy();

		Window::destroy();
		m_Window = nullptr;

		EventSystem::destroy();
		m_EventSystem = nullptr;
	}

	inline void Engine::showDebugInfo()
	{
		if (Time::time() - m_DebugTimer >= 1)
		{
			String title = String("PracticaCUDA || UPS = ").append(std::to_string(m_UPS)).append(", ").append("FPS: ").append(std::to_string(m_FPS));
			m_Window->setTitle(std::move(title));
			m_UPS = m_FPS = 0;
			m_DebugTimer = Time::time();
		}
	}

}