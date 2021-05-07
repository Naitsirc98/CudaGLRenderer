#pragma once

#include "engine/Common.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace utad
{
	class Window
	{
		friend class Engine;
		// === STATIC
	private:
		static Window* s_Instance;
	private:
		static Window* init();
		static void destroy();
	public:
		static Window& get();
		// ===
	private:
		GLFWwindow* m_Handle;
		int m_Width;
		int m_Height;
	private:
		Window(const String& title, int width = 1280, int height = 720);
	public:
		~Window();
		Window(const Window& other) = delete;
		Window& operator=(const Window& other) = delete;
		GLFWwindow* handle() const { return m_Handle; }
		int width() const { return m_Width; }
		int height() const { return m_Height; }
		void resize(int width, int height);
		void setTitle(String& title);
		bool shouldClose() const;
		void setVsync(bool vsync);
		void show();
		void pollEvents();
		void swapBuffers();
	};

}