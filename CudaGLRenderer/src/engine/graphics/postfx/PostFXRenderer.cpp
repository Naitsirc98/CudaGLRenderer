#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/postfx/CUDACommons.h"
#include "engine/graphics/postfx/InversionPostFX.cuh"
#include "engine/graphics/postfx/GammaCorrection.cuh"
#include "engine/graphics/postfx/Grayscale.cuh"
#include <cuda_gl_interop.h>

namespace utad
{
	PostFXRenderer::PostFXRenderer()
	{
	}

	PostFXRenderer::~PostFXRenderer()
	{
		delete[] m_h_ColorBuffer;
		Cuda::free(m_d_ColorBuffer);
		UTAD_DELETE(m_PixelBuffer);
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
					executeGrayscaleFX(framebufferInfo);
					break;
				case PostFX::Inversion:
					executeInversionFX(framebufferInfo);
					break;
				case PostFX::GammaCorrection:
					executeGammaCorrectionFX(framebufferInfo);
					break;
				case PostFX::Blur:
					break;
				case PostFX::Bloom:
					break;
			}
		}
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		Cuda::copyDeviceToHost(m_d_ColorBuffer, m_h_ColorBuffer, m_ColorBufferSize);
		
		Texture2DUpdateInfo updateInfo = {};
		updateInfo.format = GL_RGBA;
		updateInfo.type = GL_UNSIGNED_BYTE;
		updateInfo.level = 0;
		updateInfo.pixels = m_h_ColorBuffer;
		
		Graphics::getColorTexture()->update(std::move(updateInfo));
	}

	unsigned char* PostFXRenderer::copyTexture(int size, GLenum format, Texture2D* texture)
	{
		texture->pixels(0, format, GL_UNSIGNED_BYTE, size, m_h_ColorBuffer);
		Cuda::copyHostToDevice(m_h_ColorBuffer, m_d_ColorBuffer, size);
		return m_d_ColorBuffer;
	}

	FramebufferInfo PostFXRenderer::createFramebufferInfo()
	{
		FramebufferInfo info = {};

		Texture2D* colorTexture = Graphics::getColorTexture();

		info.width = colorTexture->width();
		info.height = colorTexture->height();
		info.pixelCount = info.width * info.height;
		info.bytes = info.pixelCount * 4 * sizeof(byte);

		if (m_h_ColorBuffer == nullptr || info.bytes != m_ColorBufferSize)
		{
			delete[] m_h_ColorBuffer;
			m_h_ColorBuffer = new unsigned char[info.bytes];
			
			Cuda::free(m_d_ColorBuffer);
			m_d_ColorBuffer = Cuda::malloc<unsigned char>(info.bytes);
		
			m_ColorBufferSize = info.bytes;
		}
		
		info.d_pixels = copyTexture(info.bytes, GL_RGBA, colorTexture);

		return info;
	}
}

//cudaGLUnmapBufferObject(m_PixelBuffer->handle());

//m_PixelBuffer->bind(GL_PIXEL_UNPACK_BUFFER);

//Graphics::getColorTexture()->bind();
//
//Texture2DUpdateInfo updateInfo = {};
//updateInfo.format = GL_BGRA;
//updateInfo.type = GL_UNSIGNED_BYTE;
//updateInfo.level = 0;
//updateInfo.pixels = nullptr;
//
//Graphics::getColorTexture()->update(std::move(updateInfo));
//
//m_PixelBuffer->unbind(GL_PIXEL_UNPACK_BUFFER);
//glFinish();




//Graphics::getColorTexture()->bind();
//
//if (m_PixelBuffer == nullptr || m_ColorBufferSize != info.bytes)
//{
//	UTAD_DELETE(m_PixelBuffer);
//
//	m_PixelBuffer = new Buffer();
//	m_PixelBuffer->bind(GL_PIXEL_PACK_BUFFER);
//
//	BufferAllocInfo allocInfo = {};
//	allocInfo.data = nullptr;
//	allocInfo.size = info.bytes;
//	allocInfo.storageFlags = GL_DYNAMIC_STORAGE_BIT;
//	
//	m_PixelBuffer->allocate(std::move(allocInfo));
//
//	cudaGLRegisterBufferObject(m_PixelBuffer->handle());
//
//	m_ColorBufferSize = info.bytes;
//}

//m_PixelBuffer->bind(GL_PIXEL_PACK_BUFFER);
//
//BufferUpdateInfo updateInfo = {};
//updateInfo.data = nullptr;
//updateInfo.size = info.bytes;
//updateInfo.offset = 0;
//
//m_PixelBuffer->update(std::move(updateInfo));

//cudaGLMapBufferObject(&info.d_pixels, m_PixelBuffer->handle());
//
//glFinish();