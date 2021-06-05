#pragma once

#include "Buffer.h"
#include "VertexArray.h"
#include "Texture.h"
#include "Framebuffer.h"
#include "Shader.h"

namespace utad
{
	class Graphics
	{
		friend class Engine;
	private:
		static Framebuffer* s_DefaultFramebuffer;
		static Texture2D* s_ColorTexture;
		static Texture2D* s_BrightnessTexture;
		static Texture2D* s_DepthTexture;
	public:
		static Framebuffer* getDefaultFramebuffer();
	private:
		static void begin();
		static void end();
		static void init();
		static void destroy();
		static void createColorTexture();
		static void createBrightnessTexture();
		static void createDepthTexture();
		static void createFramebuffer();
		static void freeFramebuffer();
	};
}