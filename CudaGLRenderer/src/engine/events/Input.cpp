#include "engine/events/Input.h"
#include "engine/events/EventSystem.h"
#include "engine/Time.h"

namespace utad
{
	Input* Input::s_Instance;

	Input* Input::init()
	{
		s_Instance = new Input();
		s_Instance->setEventCallbacks();
		return s_Instance;
	}

	void Input::destroy()
	{
		delete s_Instance;
	}

	KeyAction Input::getKey(Key key)
	{
		return s_Instance->m_Keyboard.keys[static_cast<int32_t>(key)];
	}

	bool Input::isKeyReleased(Key key)
	{
		return getKey(key) == KeyAction::Release;
	}

	bool Input::isKeyPressed(Key key)
	{
		return getKey(key) == KeyAction::Press;
	}

	bool Input::isKeyRepeated(Key key)
	{
		return getKey(key) == KeyAction::Repeat;
	}

	bool Input::isKeyTyped(Key key)
	{
		return getKey(key) == KeyAction::Type;
	}

	bool Input::isKeyActive(Key key)
	{
		return isKeyPressed(key) || isKeyRepeated(key);
	}

	MouseButtonAction Input::getMouseButton(MouseButton button)
	{
		return s_Instance->m_Mouse.buttons[static_cast<int32_t>(button)];
	}

	bool Input::isMouseButtonReleased(MouseButton button)
	{
		return getMouseButton(button) == MouseButtonAction::Release;
	}

	bool Input::isMouseButtonPressed(MouseButton button)
	{
		return getMouseButton(button) == MouseButtonAction::Press;
	}

	bool Input::isMouseButtonRepeated(MouseButton button)
	{
		return getMouseButton(button) == MouseButtonAction::Repeat;
	}

	bool Input::isMouseButtonClicked(MouseButton button)
	{
		return getMouseButton(button) == MouseButtonAction::Click;
	}

	const Vector2& Input::getMousePosition()
	{
		return s_Instance->m_Mouse.position;
	}

	const Vector2& Input::getMouseScroll()
	{
		return s_Instance->m_Mouse.scrollOffset;
	}

	void Input::setEventCallbacks()
	{
		EventSystem::addEventCallback(EventType::KeyRelease, [&](Event& e)
			{
				KeyReleaseEvent& event = static_cast<KeyReleaseEvent&>(e);
				if (isKeyPressed(event.key()))
					EventSystem::registerEvent(new KeyTypeEvent(event.key(), event.scancode(), event.keyModifiers()));
				else
					m_Keyboard.keys[static_cast<int32_t>(event.key())] = KeyAction::Release;

				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::KeyPress, [&](Event& e)
			{
				KeyPressEvent& event = static_cast<KeyPressEvent&>(e);
				m_Keyboard.keys[static_cast<int32_t>(event.key())] = KeyAction::Press;
				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::KeyRepeat, [&](Event& e)
			{
				KeyRepeatEvent& event = static_cast<KeyRepeatEvent&>(e);
				m_Keyboard.keys[static_cast<int32_t>(event.key())] = KeyAction::Repeat;
				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::KeyType, [&](Event& e)
			{
				KeyTypeEvent& event = static_cast<KeyTypeEvent&>(e);
				m_Keyboard.keys[static_cast<int32_t>(event.key())] = KeyAction::Type;
				m_Keyboard.activeModifiers = event.keyModifiers();
				EventSystem::registerEvent(new KeyReleaseEvent(event.key(), event.scancode(), event.keyModifiers()));
			});


		EventSystem::addEventCallback(EventType::MouseButtonRelease, [&](Event& e)
			{
				MouseButtonReleaseEvent& event = static_cast<MouseButtonReleaseEvent&>(e);
				if (isMouseButtonPressed(event.button()))
					EventSystem::registerEvent(new MouseButtonClickEvent(event.button(), event.keyModifiers()));
				else
					m_Mouse.buttons[static_cast<int32_t>(event.button())] = MouseButtonAction::Release;

				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::MouseButtonPress, [&](Event& e)
			{
				MouseButtonPressEvent& event = static_cast<MouseButtonPressEvent&>(e);
				m_Mouse.buttons[static_cast<int32_t>(event.button())] = MouseButtonAction::Press;
				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::MouseButtonRepeat, [&](Event& e)
			{
				MouseButtonRepeatEvent& event = static_cast<MouseButtonRepeatEvent&>(e);
				m_Mouse.buttons[static_cast<int32_t>(event.button())] = MouseButtonAction::Repeat;
				m_Keyboard.activeModifiers = event.keyModifiers();
			});

		EventSystem::addEventCallback(EventType::MouseButtonClick, [&](Event& e)
			{
				MouseButtonClickEvent& event = static_cast<MouseButtonClickEvent&>(e);
				m_Mouse.buttons[static_cast<int32_t>(event.button())] = MouseButtonAction::Click;
				m_Keyboard.activeModifiers = event.keyModifiers();
				EventSystem::registerEvent(new MouseButtonReleaseEvent(event.button(), event.keyModifiers()));
			});

		EventSystem::addEventCallback(EventType::MouseMove, [&](Event& e)
			{
				MouseMoveEvent& event = static_cast<MouseMoveEvent&>(e);
				m_Mouse.position = event.position();
			});

		EventSystem::addEventCallback(EventType::MouseScroll, [&](Event& e)
			{
				MouseScrollEvent& event = static_cast<MouseScrollEvent&>(e);
				m_Mouse.scrollOffset = event.offset();
			});
	}

	void Input::update()
	{
		Vector2& scroll = s_Instance->m_Mouse.scrollOffset;

		scroll.x = math::sign(scroll.x) * (math::abs(scroll.x) - Time::deltaTime() * 4.0f);
		scroll.y = math::sign(scroll.y) * (math::abs(scroll.y) - Time::deltaTime() * 4.0f);
	}


	bool hasKeyModifier(const KeyModifiersBitMask& keyModifiers, KeyModifier modifier)
	{
		KeyModifiersBitMask modifierBit = static_cast<KeyModifiersBitMask>(modifier);
		return (keyModifiers & modifierBit) == modifierBit;
	}

	void addKeyModifier(KeyModifiersBitMask& keyModifiers, KeyModifier modifier)
	{
		keyModifiers |= static_cast<KeyModifiersBitMask>(modifier);
	}


}