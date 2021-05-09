#pragma once

#include "Texture.h"

namespace utad
{
	class Framebuffer
	{
	private:
		Handle m_Handle;
	public:
		Framebuffer();
		Framebuffer(const Framebuffer& other) = delete;
		~Framebuffer();
		Framebuffer& operator=(const Framebuffer& other) = delete;
		Handle handle() const { return m_Handle; }
		void bind();
		void setDepthOnly();
		void addTextureAttachment(GLenum attachment, Texture2D* texture, int level = 0);
		void ensureComplete();
	public:
		static void bindDefault();
	};
}