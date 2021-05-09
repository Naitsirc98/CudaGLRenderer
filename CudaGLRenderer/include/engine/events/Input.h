#pragma once

#include "Keyboard.h"
#include "Mouse.h"

namespace utad
{
	class Input
	{
		friend class Engine;
	private:
		Keyboard m_Keyboard;
		Mouse m_Mouse;
	private:
		Input() = default;
		void setEventCallbacks();
	public:
		static KeyAction getKey(Key key);
		static bool isKeyReleased(Key key);
		static bool isKeyPressed(Key key);
		static bool isKeyRepeated(Key key);
		static bool isKeyTyped(Key key);
		static bool isKeyActive(Key key);

		static MouseButtonAction getMouseButton(MouseButton button);
		static bool isMouseButtonReleased(MouseButton button);
		static bool isMouseButtonPressed(MouseButton button);
		static bool isMouseButtonRepeated(MouseButton button);
		static bool isMouseButtonClicked(MouseButton button);

		static const Vector2& getMousePosition();
		static const Vector2& getMouseScroll();
	private:
		static Input* s_Instance;
		static Input* init();
		static void destroy();
		static void update();
	};

}