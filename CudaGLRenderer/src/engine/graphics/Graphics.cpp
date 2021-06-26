#include "engine/graphics/Graphics.h"
#include "engine/graphics/Window.h"
#include "engine/events/EventSystem.h"
#include "engine/assets/MeshPrimitives.h"
#include "engine/io/Files.h"

namespace utad
{
#define SCREEN_FRAMEBUFFER 0

	Framebuffer* Graphics::s_DefaultFramebuffer;
	Texture2D* Graphics::s_ColorTexture;
	Texture2D* Graphics::s_BrightnessTexture;
	Texture2D* Graphics::s_DepthTexture;
	Shader* Graphics::s_QuadShader;
	RenderMethod Graphics::s_RenderMethod = RenderMethod::Rasterization;


	Framebuffer* Graphics::getDefaultFramebuffer()
	{
		return s_DefaultFramebuffer;
	}

	Texture2D* Graphics::getColorTexture()
	{
		return s_ColorTexture;
	}

	Texture2D* Graphics::getBrightnessTexture()
	{
		return s_BrightnessTexture;
	}

	Texture2D* Graphics::getDepthTexture()
	{
		return s_DepthTexture;
	}

	RenderMethod Graphics::getRenderMethod()
	{
		return s_RenderMethod;
	}

	void Graphics::setRenderMethod(RenderMethod renderMethod)
	{
		s_RenderMethod = renderMethod;
	}

	void Graphics::begin()
	{
		const Window& window = Window::get();
		
		glfwMakeContextCurrent(window.handle());

		s_DefaultFramebuffer->bind();

		glViewport(0, 0, window.width(), window.height());
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	void Graphics::end()
	{
		const Window& window = Window::get();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glBlitNamedFramebuffer(
			s_DefaultFramebuffer->handle(),
			0,
			0,
			0,
			window.width(),
			window.height(),
			0,
			0,
			window.width(),
			window.height(),
			GL_COLOR_BUFFER_BIT,
			GL_LINEAR
		);
	}

	void Graphics::init()
	{
		s_RenderMethod = RenderMethod::Rasterization;

		createFramebuffer();
		createShader();

		EventSystem::addEventCallback(EventType::WindowResize, [&](const Event& e) {	
			const WindowResizeEvent& event = static_cast<const WindowResizeEvent&>(e);
			if (event.size() == Vector2i(0, 0)) return;
			freeFramebuffer();
			createFramebuffer();
		});
	}

	void Graphics::destroy()
	{
		freeFramebuffer();
		UTAD_DELETE(s_QuadShader);
	}

	void Graphics::createColorTexture()
	{
		s_ColorTexture = new Texture2D();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGBA32F;
		allocInfo.width = Window::get().width();
		allocInfo.height = Window::get().height();
		allocInfo.levels = 1;

		s_ColorTexture->wrap(GL_CLAMP_TO_EDGE);
		s_ColorTexture->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		s_ColorTexture->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		s_ColorTexture->allocate(std::move(allocInfo));
	}

	void Graphics::createBrightnessTexture()
	{
		s_BrightnessTexture = new Texture2D();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGBA32F;
		allocInfo.width = Window::get().width();
		allocInfo.height = Window::get().height();
		allocInfo.levels = 1;

		s_BrightnessTexture->wrap(GL_CLAMP_TO_EDGE);
		s_BrightnessTexture->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		s_BrightnessTexture->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		s_BrightnessTexture->allocate(std::move(allocInfo));
	}

	void Graphics::createDepthTexture()
	{
		s_DepthTexture = new Texture2D();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_DEPTH_COMPONENT24;
		allocInfo.width = Window::get().width();
		allocInfo.height = Window::get().height();
		allocInfo.levels = 1;

		s_DepthTexture->wrap(GL_CLAMP_TO_EDGE);
		s_DepthTexture->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		s_DepthTexture->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		s_DepthTexture->allocate(std::move(allocInfo));
	}

	void Graphics::createFramebuffer()
	{
		s_DefaultFramebuffer = new Framebuffer();

		createColorTexture();
		createBrightnessTexture();
		createDepthTexture();

		s_DefaultFramebuffer->addTextureAttachment(GL_COLOR_ATTACHMENT0, s_ColorTexture);
		s_DefaultFramebuffer->addTextureAttachment(GL_COLOR_ATTACHMENT1, s_BrightnessTexture);
		s_DefaultFramebuffer->addTextureAttachment(GL_DEPTH_ATTACHMENT, s_DepthTexture);

		s_DefaultFramebuffer->ensureComplete();
	}

	void Graphics::freeFramebuffer()
	{
		UTAD_DELETE(s_DefaultFramebuffer);
		UTAD_DELETE(s_ColorTexture);
		UTAD_DELETE(s_BrightnessTexture);
		UTAD_DELETE(s_DepthTexture);
	}

	void Graphics::createShader()
	{
		s_QuadShader = new Shader("Quad Shader");

		ShaderStage vertex = {};
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = Files::readAllText("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/quad.vert");

		ShaderStage fragment = {};
		fragment.type = GL_FRAGMENT_SHADER;
		fragment.sourceCode = Files::readAllText("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/quad.frag");

		s_QuadShader->attach(&vertex);
		s_QuadShader->attach(&fragment);

		s_QuadShader->compile();
	}
}