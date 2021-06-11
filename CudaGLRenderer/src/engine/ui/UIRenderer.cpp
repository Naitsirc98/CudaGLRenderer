#include "engine/ui/UIRenderer.h"
#include "engine/graphics/Window.h"

#define GLSL_VERSION "#version 330 core"

namespace utad
{
	UIRenderer* UIRenderer::s_Instance = nullptr;

	void UIRenderer::init()
	{
		s_Instance = new UIRenderer();
	}

	void UIRenderer::destroy()
	{
		UTAD_DELETE(s_Instance);
	}

	UIRenderer& UIRenderer::get()
	{
		return *s_Instance;
	}

	void UIRenderer::addUIDrawer(const UIDrawer& drawer)
	{
		m_Drawers.push_back(drawer);
	}

	void UIRenderer::removeUIDrawer(const String& name)
	{
		for (auto it = m_Drawers.begin(); it != m_Drawers.end(); ++it)
		{
			const UIDrawer& drawer = *it;
			if (drawer.name == name)
				m_Drawers.erase(it);
		}
	}

	UIRenderer::UIRenderer()
	{
		IMGUI_CHECKVERSION();
		
		m_Context = ImGui::CreateContext();

		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(Window::get().handle(), true);
		ImGui_ImplOpenGL3_Init(GLSL_VERSION);
	}

	UIRenderer::~UIRenderer()
	{
		ImGui::DestroyContext(m_Context);
		m_Drawers.clear();
	}

	void UIRenderer::render()
	{
		begin();
		{
			for (UIDrawer& drawer : m_Drawers)
				drawer.callback();
		}
		end();
	}

	void UIRenderer::begin()
	{
		ImGui::SetCurrentContext(m_Context);
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void UIRenderer::end()
	{
		ImGui::Render();
		ImGui::EndFrame();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}