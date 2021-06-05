#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/postfx/CUDACommons.h"
#include "engine/graphics/postfx/InversionPostFX.cuh"

namespace utad
{
	PostFXRenderer::PostFXRenderer()
	{
	}

	PostFXRenderer::~PostFXRenderer()
	{
		delete[] m_h_ColorBuffer;
		Cuda::free(m_d_ColorBuffer);
	}

	void PostFXRenderer::render(const RenderInfo& renderInfo)
	{
		if (renderInfo.postEffects.empty()) return;

		Graphics::getDefaultFramebuffer()->unbind();

		FramebufferInfo framebufferInfo = createFramebufferInfo();

		for(const PostFX& postFX : renderInfo.postEffects)
		{
			if (postFX == PostFX::None) continue;

			switch (postFX)
			{
				case PostFX::Grayscale:
					break;
				case PostFX::Inversion:
					executeInversionFX(framebufferInfo);
					break;
				case PostFX::ToneMapping:
					break;
				case PostFX::GammaCorrection:
					break;
				case PostFX::Blur:
					break;
				case PostFX::Bloom:
					break;
			}
		}

		Cuda::copyDeviceToHost(m_d_ColorBuffer, m_h_ColorBuffer, m_ColorBufferSize);

		Texture2DUpdateInfo updateInfo = {};
		updateInfo.format = GL_RGBA;
		updateInfo.type = GL_FLOAT;
		updateInfo.level = 0;
		updateInfo.pixels = m_h_ColorBuffer;

		Graphics::getColorTexture()->update(std::move(updateInfo));
	}

	float* PostFXRenderer::copyTexture(int size, GLenum format, Texture2D* texture)
	{
		texture->pixels(0, format, GL_FLOAT, size, m_h_ColorBuffer);
		Cuda::copyHostToDevice(m_h_ColorBuffer, m_d_ColorBuffer, size);
		return m_d_ColorBuffer;
	}

	FramebufferInfo PostFXRenderer::createFramebufferInfo()
	{
		FramebufferInfo info = {};

		Texture2D* colorTexture = Graphics::getColorTexture();

		info.width = colorTexture->width();
		info.height = colorTexture->height();
		info.size = info.width * info.height * 4 * sizeof(float);

		if (m_h_ColorBuffer == nullptr || info.size > m_ColorBufferSize)
		{
			delete[] m_h_ColorBuffer;
			m_h_ColorBuffer = new float[info.size];
			
			Cuda::free(m_d_ColorBuffer);
			m_d_ColorBuffer = Cuda::malloc<float>(info.size);

			m_ColorBufferSize = info.size;
		}

		info.d_color = copyTexture(info.size, GL_RGBA, colorTexture);

		return info;
	}
}