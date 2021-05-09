#include "engine/graphics/Window.h"
#include "engine/events/EventSystem.h"

namespace utad
{
	Window* Window::s_Instance = nullptr;

	Window* Window::init()
	{
		return s_Instance = new Window("CUDA + OpenGL");
	}

	void Window::destroy()
	{
		UTAD_DELETE(s_Instance);
	}

	Window::Window(const String& title, int width, int height) 
		: m_Handle(nullptr), m_Width(width), m_Height(height)
	{
		if (!glfwInit()) throw UTAD_EXCEPTION("Failed to init GLFW");

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	
		m_Handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		if (m_Handle == nullptr) throw UTAD_EXCEPTION("Failed to create window");

		glfwMakeContextCurrent(m_Handle);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) throw UTAD_EXCEPTION("Failed to initialize GLAD");

		setEventCallbacks();
	}

	Window::~Window()
	{
		glfwDestroyWindow(m_Handle);
		m_Handle = nullptr;
	}

	void Window::resize(int width, int height)
	{
		glfwSetWindowSize(m_Handle, width, height);
	}

	void Window::setTitle(String&& title)
	{
		glfwSetWindowTitle(m_Handle, title.c_str());
	}

	bool Window::shouldClose() const
	{
		return glfwWindowShouldClose(m_Handle);
	}

	void Window::setVsync(bool vsync)
	{
		glfwSwapInterval(vsync ? 1 : 0);
	}

	void Window::show()
	{
		glfwShowWindow(m_Handle);
	}

	void Window::swapBuffers()
	{
		glfwSwapBuffers(m_Handle);
	}

	static void keyCallback(GLFWwindow* window, int glfwKey, int scancode, int action, int glfwMod)
	{
		Key key = static_cast<Key>(glfwKey);
		KeyModifiersBitMask modifiers = static_cast<KeyModifiersBitMask>(glfwMod);

		KeyEvent* event;

		switch (action)
		{
		case GLFW_RELEASE:
			event = new KeyReleaseEvent(key, scancode, modifiers);
			break;
		case GLFW_PRESS:
			event = new KeyPressEvent(key, scancode, modifiers);
			break;
		case GLFW_REPEAT:
			event = new KeyRepeatEvent(key, scancode, modifiers);
			break;
		}

		EventSystem::registerEvent(event);
	}

	static void mouseButtonCallback(GLFWwindow* window, int glfwButton, int action, int mods)
	{
		MouseButton button = static_cast<MouseButton>(glfwButton);
		KeyModifiersBitMask modifiers = static_cast<KeyModifiersBitMask>(mods);

		MouseButtonEvent* event;

		switch (action)
		{
		case GLFW_RELEASE:
			event = new MouseButtonReleaseEvent(button, modifiers);
			break;
		case GLFW_PRESS:
			event = new MouseButtonPressEvent(button, modifiers);
			break;
		case GLFW_REPEAT:
			event = new MouseButtonRepeatEvent(button, modifiers);
			break;
		}

		EventSystem::registerEvent(event);
	}

	static void mouseMoveCallback(GLFWwindow* window, double x, double y)
	{
		EventSystem::registerEvent(new MouseMoveEvent({ (float)x, (float)y }));
	}

	static void mouseScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
	{
		EventSystem::registerEvent(new MouseScrollEvent({ (float)xOffset, (float)yOffset }));
	}

	static void mouseEnterCallback(GLFWwindow* window, int entered)
	{
		if (entered == GLFW_TRUE)
			EventSystem::registerEvent(new MouseEnterEvent());
		else
			EventSystem::registerEvent(new MouseExitEvent());
	}

	static void windowCloseCallback(GLFWwindow* window)
	{
		EventSystem::registerEvent(new WindowCloseEvent());
	}

	static void windowFocusCallback(GLFWwindow* window, int focused)
	{
		EventSystem::registerEvent(new WindowFocusEvent(focused == GLFW_TRUE));
	}

	static void windowPosCallback(GLFWwindow* window, int x, int y)
	{
		EventSystem::registerEvent(new WindowMoveEvent({ x, y }));
	}

	static void windowSizeCallback(GLFWwindow* window, int width, int height)
	{
		EventSystem::registerEvent(new WindowResizeEvent({ width, height }));
		if (width == 0 && height == 0)
			EventSystem::registerEvent(new WindowMinimizedEvent());
	}

	static void windowMaximizeCallback(GLFWwindow* window, int maximized)
	{
		EventSystem::registerEvent(new WindowMaximizedEvent(maximized == GLFW_TRUE));
	}

	void Window::setEventCallbacks()
	{
		glfwSetKeyCallback(m_Handle, keyCallback);
		glfwSetMouseButtonCallback(m_Handle, mouseButtonCallback);
		glfwSetCursorPosCallback(m_Handle, mouseMoveCallback);
		glfwSetScrollCallback(m_Handle, mouseScrollCallback);
		glfwSetCursorEnterCallback(m_Handle, mouseEnterCallback);
		glfwSetWindowCloseCallback(m_Handle, windowCloseCallback);
		glfwSetWindowFocusCallback(m_Handle, windowFocusCallback);
		glfwSetWindowPosCallback(m_Handle, windowPosCallback);
		glfwSetWindowSizeCallback(m_Handle, windowSizeCallback);
		glfwSetWindowMaximizeCallback(m_Handle, windowMaximizeCallback);

		EventSystem::addEventCallback(EventType::WindowMaximized, [&](Event& e)
			{
				WindowMaximizedEvent& event = static_cast<WindowMaximizedEvent&>(e);
				if (event.maximized())
					m_State = WindowState::MAXIMIZED;
				else
					m_State = WindowState::WINDOWED;
			});

		EventSystem::addEventCallback(EventType::WindowMinimized, [&](Event& e)
			{
				m_State = WindowState::MINIMIZED;
			});

		EventSystem::addEventCallback(EventType::WindowResize, [&](Event& e)
			{
				WindowResizeEvent& event = static_cast<WindowResizeEvent&>(e);
				if (m_State == WindowState::MINIMIZED && (event.size().x > 0 || event.size().y > 0))
					m_State = WindowState::WINDOWED;
			});

	}
}