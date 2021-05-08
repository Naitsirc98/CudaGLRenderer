#include "engine/graphics/Window.h"

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

	void Window::pollEvents()
	{
		glfwPollEvents();
	}

	void Window::swapBuffers()
	{
		glfwSwapBuffers(m_Handle);
	}
}