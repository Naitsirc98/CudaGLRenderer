#include "engine/Engine.h"
#include "engine/Time.h"
#include <iostream>

namespace utad
{
	static const float TARGET_DELAY = 1.0f / 60.0f;

	void Engine::launch()
	{
		static Engine engine;
		if (engine.m_Active) return;

		engine.start();
		engine.run();
		engine.shutdown();
	}

	Engine::Engine()
	{
		shutdown();
	}

	void Engine::start()
	{
		m_Window = Window::init();

		m_Window->show();
	}

	void Engine::run()
	{
		const Window* window = m_Window;
		float lastFrame = 0;
		int fps = 0;
		float debugTimer = Time::time();

		while (!window->shouldClose())
		{
			const float now = Time::time();
			const float deltaTime = now - lastFrame;
			lastFrame = now;
			Time::s_DeltaTime = deltaTime;

			update(deltaTime);

			render();
			++fps;

			if (Time::time() - debugTimer >= 1)
			{
				String title = String("PracticaCUDA || UPS = ").append(std::to_string(m_UPS)).append(", ").append("FPS: ").append(std::to_string(fps));
				m_Window->setTitle(std::move(title));
				m_UPS = fps = 0;
				debugTimer = Time::time();
			}
		}
	}

	void Engine::update(float deltaTime)
	{
		m_UpdateDelay += deltaTime;

		while (m_UpdateDelay >= TARGET_DELAY)
		{
			m_Window->pollEvents();

			// UPDATE
			m_UpdateDelay -= TARGET_DELAY;
			++m_UPS;
		}
	}

	void Engine::render()
	{
		glClear(GL_COLOR_BUFFER_BIT);
		m_Window->swapBuffers();
	}

	void Engine::shutdown()
	{
		if (m_Window != nullptr)
		{
			Window::destroy();
			m_Window = nullptr;
		}
	}

}