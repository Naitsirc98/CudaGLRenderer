#pragma once

#include "engine/Engine.h"

namespace utad
{
	class CameraController : public Script
	{
	public:

		float normalSpeed = 3;
		Camera* camera;

		void onStart() override
		{
			camera = &Scene::get().camera();
			camera->sensitivity(0.08f);
		}

		void checkWindowSize()
		{
			if (Input::isKeyTyped(Key::Key_F1)) Window::get().state(WindowState::WINDOWED);
			if (Input::isKeyTyped(Key::Key_F2)) Window::get().state(WindowState::MAXIMIZED);
			if (Input::isKeyTyped(Key::Key_F3)) Window::get().state(WindowState::FULLSCREEN);
		}

		void checkGameSettings()
		{
			if (Input::isKeyTyped(Key::Key_Escape))
			{
				Window& window = Window::get();
				if (window.cursorMode() == CursorMode::CAPTURED)
					window.cursorMode(CursorMode::NORMAL);
				else
					window.cursorMode(CursorMode::CAPTURED);
			}

			checkWindowSize();
		}

		void onUpdate() override
		{
			checkGameSettings();

			float speed = this->normalSpeed * Time::deltaTime();
			if (Input::isKeyActive(Key::Key_Left_Shift)) speed *= 2;
			if (Input::isKeyActive(Key::Key_Left_Alt)) speed /= 2;

			if (Input::isKeyActive(Key::Key_W)) camera->move(CameraDirection::Forward, speed);
			if (Input::isKeyActive(Key::Key_S)) camera->move(CameraDirection::Backwards, speed);
			if (Input::isKeyActive(Key::Key_A)) camera->move(CameraDirection::Left, speed);
			if (Input::isKeyActive(Key::Key_D)) camera->move(CameraDirection::Right, speed);
			if (Input::isKeyActive(Key::Key_Space)) camera->move(CameraDirection::Up, speed);
			if (Input::isKeyActive(Key::Key_Left_Control)) camera->move(CameraDirection::Down, speed);

			if (Input::isKeyActive(Key::Key_P))
			{
				std::cout << "Camera position = "
					<< camera->position().x << ", "
					<< camera->position().y << ", "
					<< camera->position().z << std::endl;

				std::cout << "Camera direction = "
					<< camera->forward().x << ", "
					<< camera->forward().y << ", "
					<< camera->forward().z << std::endl;
			}

			if (Window::get().cursorMode() == CursorMode::CAPTURED)
			{
				camera->lookAt(Input::getMousePosition());
				camera->zoom(Input::getMouseScroll().y);
			}
		}
	};
}