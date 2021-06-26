#pragma once

#include "Buffer.h"
#include "VertexArray.h"
#include "Texture.h"
#include "Framebuffer.h"
#include "Shader.h"

namespace utad
{
	enum class RenderMethod
	{
		Rasterization,
		RayTracing
	};

	class Graphics
	{
		friend class Engine;
	private:
		static Framebuffer* s_DefaultFramebuffer;
		static Texture2D* s_ColorTexture;
		static Texture2D* s_BrightnessTexture;
		static Texture2D* s_DepthTexture;
		static Shader* s_QuadShader;
		static RenderMethod s_RenderMethod;
	public:
		static Framebuffer* getDefaultFramebuffer();
		static Texture2D* getColorTexture();
		static Texture2D* getBrightnessTexture();
		static Texture2D* getDepthTexture();
		static RenderMethod getRenderMethod();
		static void setRenderMethod(RenderMethod renderMethod);
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
		static void createShader();
	};
}