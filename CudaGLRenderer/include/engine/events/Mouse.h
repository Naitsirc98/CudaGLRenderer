#pragma once

#include "engine/Common.h"

namespace utad
{
	enum class MouseButton
	{
		Mouse_Button_1 = 0,
		Mouse_Button_2 = 1,
		Mouse_Button_3 = 2,
		Mouse_Button_4 = 3,
		Mouse_Button_5 = 4,
		Mouse_Button_6 = 5,
		Mouse_Button_7 = 6,
		Mouse_Button_8 = 7,
		Mouse_Button_Last = Mouse_Button_8,
		Mouse_Button_Left = Mouse_Button_1,
		Mouse_Button_Right = Mouse_Button_2,
		Mouse_Button_Middle = Mouse_Button_3
	};

	const uint NUMBER_OF_MOUSE_BUTTONS = static_cast<uint>(MouseButton::Mouse_Button_Last);

	enum class MouseButtonAction
	{
		Release = 0,
		Press = 1,
		Repeat = 2,
		Click = 3
	};

	struct Mouse
	{
		friend class Input;
	public:
		Vector2 position;
		Vector2 scrollOffset;
		MouseButtonAction buttons[NUMBER_OF_MOUSE_BUTTONS];
	private:
		Mouse() : position({0, 0}), scrollOffset({0, 0})
		{
			memset(buttons, static_cast<int32_t>(MouseButtonAction::Release), NUMBER_OF_MOUSE_BUTTONS * sizeof(MouseButtonAction));
		}
	};
}