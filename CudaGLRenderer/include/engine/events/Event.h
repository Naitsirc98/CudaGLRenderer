#pragma once
#include "engine/Common.h"

namespace utad
{
	enum class EventType
	{
		KeyRelease,
		KeyPress,
		KeyRepeat,
		KeyType,
		MouseEnter,
		MouseExit,
		MouseMove,
		MouseButtonRelease,
		MouseButtonPress,
		MouseButtonRepeat,
		MouseButtonClick,
		MouseScroll,
		WindowClose,
		WindowFocus,
		WindowMove,
		WindowResize,
		WindowMaximized,
		WindowMinimized,
		ApplicationExit,
		UserEvent
	};

	class Event
	{
	private:
		bool m_Consumed;
	public:
		Event() : m_Consumed(false) {};
		virtual ~Event() = default;
		virtual EventType type() const = 0;
		bool consumed() const { return m_Consumed; }
		void consume() { m_Consumed = true; }
	};

	using EventCallback = Function<void, Event&>;
}
