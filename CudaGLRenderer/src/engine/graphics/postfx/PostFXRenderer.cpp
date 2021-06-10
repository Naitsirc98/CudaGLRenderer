#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/postfx/InversionFX.cuh"
#include "engine/graphics/postfx/GammaCorrectionFX.cuh"
#include "engine/graphics/postfx/GrayscaleFX.cuh"
#include "engine/graphics/postfx/GaussianBlurFX.cuh"
#include <cuda_gl_interop.h>

#define NUM_CHANNELS 4

namespace utad
{
	PostFXRenderer::PostFXRenderer()
	{
		size_t count = static_cast<size_t>(PostFX::_MaxEnumValue);

		m_PostFXExecutors.resize(count);

		m_PostFXExecutors[static_cast<size_t>(PostFX::Grayscale)] = new GrayscaleFX();
		m_PostFXExecutors[static_cast<size_t>(PostFX::GammaCorrection)] = new GammaCorrectionFX();
		m_PostFXExecutors[static_cast<size_t>(PostFX::Inversion)] = new InversionFX();
		m_PostFXExecutors[static_cast<size_t>(PostFX::Blur)] = new GaussianBlurFX();
	}

	PostFXRenderer::~PostFXRenderer()
	{
		for (PostFXExecutor* executor : m_PostFXExecutors)
			delete executor;

		delete[] m_h_ColorBuffer;

		Cuda::free(m_d_ColorBuffer);
		
		UTAD_DELETE(m_PixelBuffer);
	}

	void PostFXRenderer::render(const SceneSetup& scene)
	{
		if (scene.postEffects.empty()) return;

		begin(scene);

		for(PostFX postFX : scene.postEffects)
		{
			if (postFX == PostFX::_MaxEnumValue) continue;
			PostFXExecutor* executor = m_PostFXExecutors[static_cast<size_t>(postFX)];
			executor->execute(m_PostFXInfo);
		}

		end(scene);
	}

	void PostFXRenderer::begin(const SceneSetup& scene)
	{
		Graphics::getDefaultFramebuffer()->unbind();
		PostFXInfo& info = m_PostFXInfo;
		Texture2D* colorTexture = Graphics::getColorTexture();

		info.width = colorTexture->width();
		info.height = colorTexture->height();
		info.exposure = scene.camera.exposure();

		const size_t size = info.width * info.height * NUM_CHANNELS;

		if (m_h_ColorBuffer == nullptr || size != m_ColorBufferSize)
		{
			delete[] m_h_ColorBuffer;
			m_h_ColorBuffer = new unsigned char[size];
			
			Cuda::free(m_d_ColorBuffer);
			m_d_ColorBuffer = (unsigned char*)Cuda::malloc(size);
		
			m_ColorBufferSize = size;
		}
		
		info.d_pixels = copyTexture(size, GL_RGBA, colorTexture);
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