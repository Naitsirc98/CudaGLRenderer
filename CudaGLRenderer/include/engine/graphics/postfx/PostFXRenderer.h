#pragma once

namespace utad
{
	struct RenderInfo;
	struct FramebufferInfo;

	enum class PostFX
	{
		None,
		Grayscale,
		Inversion,
		ToneMapping,
		GammaCorrection,
		Blur,
		Bloom
	};

	class PostFXRenderer
	{
		friend class Scene;
	private:
		float* m_h_ColorBuffer{nullptr};
		float* m_d_ColorBuffer{nullptr};
		int m_ColorBufferSize{0};
	private:
		PostFXRenderer();
		~PostFXRenderer();
		void render(const RenderInfo& renderInfo);
		FramebufferInfo createFramebufferInfo();
		float* copyTexture(int size, GLenum format, Texture2D* texture);
	};
}