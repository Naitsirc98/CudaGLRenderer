#pragma once

#include "engine/Common.h"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

namespace utad
{
	struct UIDrawer
	{
		String name;
		Function<void> callback;

		bool operator==(const UIDrawer& other) noexcept
		{
			return name == other.name;
		}
	};

	class UIRenderer
	{
		friend class Engine;
	private:
		static UIRenderer* s_Instance;
	private:
		static void init();
		static void destroy();
	public:
		static UIRenderer& get();
	private:
		ImGuiContext* m_Context{nullptr};
		LinkedList<UIDrawer> m_Drawers;
	public:
		void addUIDrawer(const UIDrawer& drawer);
		void removeUIDrawer(const String& name);
	private:
		UIRenderer();
		~UIRenderer();
		void render();
		void begin();
		void end();
	};
}
