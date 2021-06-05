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
		static VertexArray* s_QuadVAO;
		static Shader* s_QuadShader;
	public:
		static Framebuffer* getDefaultFramebuffer();
		static Texture2D* getColorTexture();
		static Texture2D* getBrightnessTexture();
		static Texture2D* getDepthTexture();
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
		static void createQuad();
		static void createShader();
	};
}