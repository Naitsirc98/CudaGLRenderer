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
	private:
		PostFXRenderer();
		~PostFXRenderer();
		void render(const RenderInfo& renderInfo);
		FramebufferInfo createFramebufferInfo();
		unsigned char* copyTexture(int size, GLenum format, Texture2D* texture);
	};
}