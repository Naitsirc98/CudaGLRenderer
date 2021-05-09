#pragma once
#include "engine/events/Event.h"

namespace utad
{
	class WindowCloseEvent : public Event
	{
	public:
		WindowCloseEvent() = default;
		~WindowCloseEvent() = default;
		EventType type() const override { return EventType::WindowClose; }
	};

	class WindowFocusEvent : public Event
	{
	private:
		bool m_Focused;
	public:
		explicit WindowFocusEvent(bool focused) : m_Focused(focused) {}
		~WindowFocusEvent() = default;
		bool focused() const { return m_Focused; }
		EventType type() const override { return EventType::WindowFocus; }
	};

	class WindowMoveEvent : public Event
	{
	private:
		Vector2i m_Position;
	public:
		explicit WindowMoveEvent(const Vector2i& position) : m_Position(position) {}
		~WindowMoveEvent() = default;
		const Vector2i& position() const { return m_Position; }
		EventType type() const override { return EventType::WindowMove; }
	};

	class WindowResizeEvent : public Event
	{
	private:
		Vector2i m_Size;
	public:
		explicit WindowResizeEvent(const Vector2i& size) : m_Size(size) {}
		~WindowResizeEvent() = default;
		const Vector2i& size() const { return m_Size; }
		EventType type() const override { return EventType::WindowResize; }
	};

	class WindowMaximizedEvent : public Event
	{
	private:
		bool m_Maximized;
	public:
		explicit WindowMaximizedEvent(bool maximized) : m_Maximized(maximized) {}
		~WindowMaximizedEvent() = default;
		bool maximized() const { return m_Maximized; }
		EventType type() const override { return EventType::WindowMaximized; }
	};

	class WindowMinimizedEvent : public Event
	{
	public:
		WindowMinimizedEvent() = default;
		~WindowMinimizedEvent() = default;
		EventType type() const override { return EventType::WindowMinimized; }
	};

	class ApplicationExitEvent : public Event
	{
	public:
		ApplicationExitEvent() = default;
		~ApplicationExitEvent() = default;
		EventType type() const override { return EventType::ApplicationExit; }
	};
}