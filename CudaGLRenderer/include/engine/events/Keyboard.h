#pragma once

#include "engine/Common.h"

namespace utad
{
	enum class Key
	{
		Key_Unknown = -1,
		/* Printable keys */
		Key_Space = 32,
		Key_Apostrophe = 39, /* ' */
		Key_Comma = 44, /* , */
		Key_Minus = 45, /* - */
		Key_Period = 46, /* . */
		Key_Slash = 47, /* / */
		Key_0 = 48,
		Key_1 = 49,
		Key_2 = 50,
		Key_3 = 51,
		Key_4 = 52,
		Key_5 = 53,
		Key_6 = 54,
		Key_7 = 55,
		Key_8 = 56,
		Key_9 = 57,
		Key_Semicolon = 59, /* ; */
		Key_Equal = 61, /* = */
		Key_A = 65,
		Key_B = 66,
		Key_C = 67,
		Key_D = 68,
		Key_E = 69,
		Key_F = 70,
		Key_G = 71,
		Key_H = 72,
		Key_I = 73,
		Key_J = 74,
		Key_K = 75,
		Key_L = 76,
		Key_M = 77,
		Key_N = 78,
		Key_O = 79,
		Key_P = 80,
		Key_Q = 81,
		Key_R = 82,
		Key_S = 83,
		Key_T = 84,
		Key_U = 85,
		Key_V = 86,
		Key_W = 87,
		Key_X = 88,
		Key_Y = 89,
		Key_Z = 90,
		Key_Left_Bracket = 91, /* [ */
		Key_Backslash = 92, /* \ */
		Key_Right_Bracket = 93, /* ] */
		Key_Grave_Accent = 96, /* ` */
		Key_World_1 = 161, /* non-US #1 */
		Key_World_2 = 162, /* non-US #2 */

		/* Function keys */
		Key_Escape = 256,
		Key_Enter = 257,
		Key_Tab = 258,
		Key_Backspace = 259,
		Key_Insert = 260,
		Key_Delete = 261,
		Key_Right = 262,
		Key_Left = 263,
		Key_Down = 264,
		Key_Up = 265,
		Key_Page_up = 266,
		Key_Page_down = 267,
		Key_Home = 268,
		Key_End = 269,
		Key_Caps_lock = 280,
		Key_Scroll_lock = 281,
		Key_Num_lock = 282,
		Key_Print_screen = 283,
		Key_Pause = 284,
		Key_F1 = 290,
		Key_F2 = 291,
		Key_F3 = 292,
		Key_F4 = 293,
		Key_F5 = 294,
		Key_F6 = 295,
		Key_F7 = 296,
		Key_F8 = 297,
		Key_F9 = 298,
		Key_F10 = 299,
		Key_F11 = 300,
		Key_F12 = 301,
		Key_F13 = 302,
		Key_F14 = 303,
		Key_F15 = 304,
		Key_F16 = 305,
		Key_F17 = 306,
		Key_F18 = 307,
		Key_F19 = 308,
		Key_F20 = 309,
		Key_F21 = 310,
		Key_F22 = 311,
		Key_F23 = 312,
		Key_F24 = 313,
		Key_F25 = 314,
		Key_Kp_0 = 320,
		Key_Kp_1 = 321,
		Key_Kp_2 = 322,
		Key_Kp_3 = 323,
		Key_Kp_4 = 324,
		Key_Kp_5 = 325,
		Key_Kp_6 = 326,
		Key_Kp_7 = 327,
		Key_Kp_8 = 328,
		Key_Kp_9 = 329,
		Key_Kp_Decimal = 330,
		Key_Kp_Divide = 331,
		Key_Kp_Multiply = 332,
		Key_Kp_Subtract = 333,
		Key_Kp_Add = 334,
		Key_Kp_Enter = 335,
		Key_Kp_Equal = 336,
		Key_Left_Shift = 340,
		Key_Left_Control = 341,
		Key_Left_Alt = 342,
		Key_Left_Super = 343,
		Key_Right_Shift = 344,
		Key_Right_Control = 345,
		Key_Right_Alt = 346,
		Key_Right_Super = 347,
		Key_Menu = 348,
		Key_Last = Key_Menu
	};

	const uint NUMBER_OF_KEYS = static_cast<uint>(Key::Key_Last);

	enum class KeyModifier
	{
		Mod_Shift = 0x0001,
		Mod_Control = 0x0002,
		Mod_Alt = 0x0004,
		Mod_Super = 0x0008,
		Mod_Caps_Lock = 0x0010,
		Mod_Num_Lock = 0x0020
	};

	bool hasKeyModifier(const KeyModifiersBitMask& keyModifiers, KeyModifier modifier);
	void addKeyModifier(KeyModifiersBitMask& keyModifiers, KeyModifier modifier);

	enum class KeyAction
	{
		Release = 0,
		Press = 1,
		Repeat = 2,
		Type = 3
	};

	class Keyboard
	{
		friend class Input;
	public:
		KeyAction keys[NUMBER_OF_KEYS];
		KeyModifiersBitMask activeModifiers;
	private:
		Keyboard()
		{
			memset(keys, static_cast<int>(KeyAction::Release), NUMBER_OF_KEYS * sizeof(KeyAction));
			activeModifiers = 0;
		}
	};
}