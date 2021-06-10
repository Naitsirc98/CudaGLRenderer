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
		Bloom,
		_MaxEnumValue
	};

	const uint PostFXCount = 5;

	class PostFXRenderer
	{
		friend class Scene;
	private:
		unsigned char* m_h_ColorBuffer{nullptr};
		unsigned char* m_d_ColorBuffer{nullptr};
		int m_ColorBufferSize{0};
		Buffer* m_PixelBuffer{nullptr};
		CudaResource* m_TextureResource{nullptr};
		PostFXInfo m_PostFXInfo;
		// PostFX Executors
		ArrayList<PostFXExecutor*> m_PostFXExecutors;
	private:
		PostFXRenderer();
		~PostFXRenderer();
		void render(const SceneSetup& renderInfo);
		void begin(const SceneSetup& scene);
		void end(const SceneSetup& scene);
		unsigned char* copyTexture(int size, GLenum format, Texture2D* texture);
	};
}