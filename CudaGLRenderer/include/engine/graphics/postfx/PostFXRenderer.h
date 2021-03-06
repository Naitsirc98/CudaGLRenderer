#pragma once

#include "GrayscaleFX.cuh"
#include "InversionFX.cuh"
#include "GaussianBlurFX.cuh"
#include "GammaCorrectionFX.cuh"

namespace utad
{
	struct SceneSetup;
	struct RenderInfo;

	enum class PostFX
	{
		Grayscale,
		Inversion,
		GammaCorrection,
		Blur,
		Sharpening,
		EdgeDetection,
		Emboss,
		_MaxEnumValue
	};

	const uint PostFXCount = static_cast<uint>(PostFX::_MaxEnumValue);
	const Array<String, 7> PostFXNames = {
		"Grayscale",
		"Inversion",
		"GammaCorrection",
		"Blur",
		"Sharpening",
		"EdgeDetection",
		"Emboss"
	};

	class PostFXRenderer
	{
		friend class Scene;
	private:
		int m_ColorBufferSize{0};
		CudaResource m_ColorTexture{nullptr};
		CudaSurface m_ColorBuffer{NULL};
		PostFXInfo m_PostFXInfo;
		// PostFX Executors
		ArrayList<PostFXExecutor*> m_PostFXExecutors;
	private:
		PostFXRenderer();
		~PostFXRenderer();
		void render(const SceneSetup& renderInfo);
		void begin(const SceneSetup& scene);
		void end(const SceneSetup& scene);
		void bindTextureToSurface(Texture2D* texture, size_t size);
		void recreateCudaResources(Texture2D* texture, size_t size);
	};
}