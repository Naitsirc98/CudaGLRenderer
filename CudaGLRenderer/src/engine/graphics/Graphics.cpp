#include "engine/graphics/Graphics.h"
#include "engine/graphics/Window.h"
#include "engine/events/EventSystem.h"

namespace utad
{
#define SCREEN_FRAMEBUFFER 0

	Framebuffer* Graphics::s_DefaultFramebuffer;
	Texture2D* Graphics::s_ColorTexture;
	Texture2D* Graphics::s_BrightnessTexture;
	Texture2D* Graphics::s_DepthTexture;


	Framebuffer* Graphics::getDefaultFramebuffer()
	{
		return s_DefaultFramebuffer;
	}

	void Graphics::begin()
	{
		s_DefaultFramebuffer->bind();

		const Window& window = Window::get();
		glViewport(0, 0, window.width(), window.height());
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	void Graphics::end()
	{
		const GLint width = Window::get().width();
		const GLint height = Window::get().height();

		glBlitNamedFramebuffer(
			s_DefaultFramebuffer->handle(),
			SCREEN_FRAMEBUFFER,
			0,
			0,
			width,
			height,
			0,
			0,
			width,
			height,
			GL_COLOR_BUFFER_BIT,
			GL_LINEAR);

		Window::get().swapBuffers();
	}

	void Graphics::init()
	{
		createFramebuffer();

		EventSystem::addEventCallback(EventType::WindowResize, [&](const Event& e) {			
			freeFramebuffer();
			createFramebuffer();
		});
	}

	void Graphics::destroy()
	{
		freeFramebuffer();
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
}