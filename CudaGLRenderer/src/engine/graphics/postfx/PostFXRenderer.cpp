#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/postfx/CUDACommons.h"
#include "engine/graphics/postfx/InversionPostFX.cuh"
#include "engine/graphics/postfx/GammaCorrection.cuh"
#include "engine/graphics/postfx/Grayscale.cuh"
#include "engine/graphics/postfx/FiltersFX.cuh"
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

	void PostFXRenderer::render(const SceneSetup& scene)
	{
		if (scene.postEffects.empty()) return;

		begin(scene);

		for(const PostFX& postFX : scene.postEffects)
		{
			if (postFX == PostFX::None) continue;

			switch (postFX)
			{
				case PostFX::Grayscale:
					executeGrayscaleFX(m_RenderInfo);
					break;
				case PostFX::Inversion:
					executeInversionFX(m_RenderInfo);
					break;
				case PostFX::GammaCorrection:
					executeGammaCorrectionFX(m_RenderInfo);
					break;
				case PostFX::Blur:
					executeBlurFX(m_RenderInfo);
					break;
				case PostFX::Bloom:
					break;
			}
		}

		end(scene);
	}

	void PostFXRenderer::begin(const SceneSetup& scene)
	{
		Graphics::getDefaultFramebuffer()->unbind();
		RenderInfo& info = m_RenderInfo;
		Texture2D* colorTexture = Graphics::getColorTexture();

		info.width = colorTexture->width();
		info.height = colorTexture->height();
		info.pixelCount = info.width * info.height;
		info.bytes = info.pixelCount * 4 * sizeof(byte);
		info.exposure = scene.camera.exposure();

		if (m_h_ColorBuffer == nullptr || info.bytes != m_ColorBufferSize)
		{
			delete[] m_h_ColorBuffer;
			m_h_ColorBuffer = new unsigned char[info.bytes];
			
			Cuda::free(m_d_ColorBuffer);
			m_d_ColorBuffer = (unsigned char*)Cuda::malloc(info.bytes);
		
			m_ColorBufferSize = info.bytes;
		}
		
		info.d_pixels = copyTexture(info.bytes, GL_RGBA, colorTexture);
	}

	void PostFXRenderer::end(const SceneSetup& scene)
	{
		cudaDeviceSynchronize();
		CUDA_CHECK;

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