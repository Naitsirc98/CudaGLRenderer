#pragma once
#include "Event.h"
#include "engine/events/Mouse.h"

namespace utad
{
	class MouseButtonEvent : public Event
	{
	private:
		MouseButton m_Button;
		KeyModifiersBitMask m_KeyModifiers;
	public:
		MouseButtonEvent(MouseButton button, KeyModifiersBitMask modifiers)
			: m_Button(button), m_KeyModifiers(modifiers) {}
		virtual ~MouseButtonEvent() = default;
		MouseButton button() const { return m_Button; }
		KeyModifiersBitMask keyModifiers() const { return m_KeyModifiers; }
		virtual MouseButtonAction action() const = 0;
	};

	class MouseButtonReleaseEvent : public MouseButtonEvent
	{
	public:
		MouseButtonReleaseEvent(MouseButton button, KeyModifiersBitMask modifiers)
			: MouseButtonEvent(button, modifiers) {}
		~MouseButtonReleaseEvent() = default;
		MouseButtonAction action() const override { return MouseButtonAction::Release; }
		EventType type() const override { return EventType::MouseButtonRelease; }
	};

	class MouseButtonPressEvent : public MouseButtonEvent
	{
	public:
		MouseButtonPressEvent(MouseButton button, KeyModifiersBitMask modifiers)
			: MouseButtonEvent(button, modifiers) {}
		~MouseButtonPressEvent() = default;
		MouseButtonAction action() const override { return MouseButtonAction::Press; }
		EventType type() const override { return EventType::MouseButtonPress; }
	};

	class MouseButtonRepeatEvent : public MouseButtonEvent
	{
	public:
		MouseButtonRepeatEvent(MouseButton button, KeyModifiersBitMask modifiers)
			: MouseButtonEvent(button, modifiers) {}
		~MouseButtonRepeatEvent() = default;
		MouseButtonAction action() const override { return MouseButtonAction::Repeat; }
		EventType type() const override { return EventType::MouseButtonRepeat; }
	};

	class MouseButtonClickEvent : public MouseButtonEvent
	{
	public:
		MouseButtonClickEvent(MouseButton button, KeyModifiersBitMask modifiers)
			: MouseButtonEvent(button, modifiers) {}
		~MouseButtonClickEvent() = default;
		MouseButtonAction action() const override { return MouseButtonAction::Click; }
		EventType type() const override { return EventType::MouseButtonClick; }
	};

	class MouseMoveEvent : public Event
	{
	private:
		Vector2 m_Position;
	public:
		explicit MouseMoveEvent(const Vector2& position) : m_Position(position) {}
		~MouseMoveEvent() = default;
		const Vector2& position() const { return m_Position; }
		EventType type() const override { return EventType::MouseMove; }
	};

	class MouseEnterEvent : public Event
	{
	public:
		MouseEnterEvent() = default;
		~MouseEnterEvent() = default;
		EventType type() const override { return EventType::MouseEnter; }
	};

	class MouseExitEvent : public Event
	{
	public:
		MouseExitEvent() = default;
		~MouseExitEvent() = default;
		EventType type() const override { return EventType::MouseExit; }
	};

	class MouseScrollEvent : public Event
	{
	private:
		Vector2 m_Offset;
	public:
		explicit MouseScrollEvent(const Vector2& offset) : m_Offset(offset) {}
		~MouseScrollEvent() = default;
		const Vector2& offset() const { return m_Offset; }
		EventType type() const override { return EventType::MouseScroll; }
	};

}