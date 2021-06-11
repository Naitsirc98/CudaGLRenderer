#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/graphics/postfx/PostFXRenderer.h"
#include "engine/graphics/postfx/InversionFX.cuh"
#include "engine/graphics/postfx/GammaCorrectionFX.cuh"
#include "engine/graphics/postfx/GrayscaleFX.cuh"
#include "engine/graphics/postfx/GaussianBlurFX.cuh"
#include "engine/graphics/postfx/SharpeningFX.cuh"

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
		m_PostFXExecutors[static_cast<size_t>(PostFX::Sharpening)] = new SharpeningFX();
	}

	PostFXRenderer::~PostFXRenderer()
	{
		for (PostFXExecutor* executor : m_PostFXExecutors)
			delete executor;

		Cuda::destroyResource(m_ColorTexture);
		CUDA_CALL(cudaDestroySurfaceObject(m_ColorBuffer));
	}

	void PostFXRenderer::render(const SceneSetup& scene)
	{
		if (scene.postEffects.empty()) return;

		begin(scene);

		for(PostFX postFX : scene.postEffects)
		{
			if (postFX == PostFX::_MaxEnumValue) continue;
			PostFXExecutor* executor = m_PostFXExecutors[static_cast<size_t>(postFX)];
			if (executor == nullptr) continue;
			executor->execute(m_PostFXInfo);
		}

		end(scene);
	}

	void PostFXRenderer::begin(const SceneSetup& scene)
	{
		Graphics::getDefaultFramebuffer()->unbind();
		PostFXInfo& info = m_PostFXInfo;
		Texture2D* colorTexture = Graphics::getColorTexture();
		colorTexture->bind();

		info.width = colorTexture->width();
		info.height = colorTexture->height();
		info.exposure = scene.camera.exposure();

		const size_t size = info.width * info.height * NUM_CHANNELS;
		bindTextureToSurface(colorTexture, size);
		info.colorBuffer = m_ColorBuffer;
	}

	void PostFXRenderer::end(const SceneSetup& scene)
	{
		CUDA_CALL(cudaGraphicsUnmapResources(1, &m_ColorTexture));
		Graphics::getColorTexture()->unbind();
	}

	void PostFXRenderer::bindTextureToSurface(Texture2D* texture, size_t size)
	{
		if (m_ColorTexture == nullptr || m_ColorBuffer == NULL || m_ColorBufferSize != size)
			recreateCudaResources(texture, size);
		else 
			CUDA_CALL(cudaGraphicsMapResources(1, &m_ColorTexture));
	}

	void PostFXRenderer::recreateCudaResources(Texture2D* texture, size_t size)
	{
		if(m_ColorTexture != nullptr) Cuda::destroyResource(m_ColorTexture);
		if(m_ColorBuffer != NULL) CUDA_CALL(cudaDestroySurfaceObject(m_ColorBuffer));

		Cuda::createResource(m_ColorTexture, texture->handle());

		CUDA_CALL(cudaGraphicsMapResources(1, &m_ColorTexture));

		CudaArray* contents;
		CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&contents, m_ColorTexture, 0, 0));

		CudaResourceDescription description = {};
		description.res.array.array = contents;
		description.resType = cudaResourceTypeArray;

		CUDA_CALL(cudaCreateSurfaceObject(&m_ColorBuffer, &description));
		m_ColorBufferSize = size;
	}
}