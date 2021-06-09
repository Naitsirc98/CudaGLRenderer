#pragma once

#include "CUDACommons.h"

namespace utad
{
	struct SceneSetup;
	struct RenderInfo;

	enum class PostFX
	{
		None,
		Grayscale,
		Inversion,
		GammaCorrection,
		Blur,
		Bloom
	};

	class PostFXRenderer
	{
		friend class Scene;
	private:
		unsigned char* m_h_ColorBuffer{nullptr};
		unsigned char* m_d_ColorBuffer{nullptr};
		int m_ColorBufferSize{0};
		Buffer* m_PixelBuffer{nullptr};
		CudaResource* m_TextureResource{ nullptr };
		RenderInfo m_RenderInfo;
	private:
		PostFXRenderer();
		~PostFXRenderer();
		void render(const SceneSetup& renderInfo);
		void begin();
		void end();
		unsigned char* copyTexture(int size, GLenum format, Texture2D* texture);
	};
}