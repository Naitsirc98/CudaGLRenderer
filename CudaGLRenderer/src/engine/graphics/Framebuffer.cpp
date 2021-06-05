#include "engine/graphics/Framebuffer.h"
#include "engine/Common.h"

namespace utad
{
	Framebuffer::Framebuffer()
	{
		glCreateFramebuffers(1, &m_Handle);
	}

	Framebuffer::~Framebuffer()
	{
		glDeleteFramebuffers(1, &m_Handle);
		m_Handle = NULL;
	}

	void Framebuffer::bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_Handle);
	}

	void Framebuffer::unbind()
	{
		Framebuffer::bindDefault();
	}

	void Framebuffer::setDepthOnly()
	{
		GLenum buffer = GL_NONE;
		glNamedFramebufferReadBuffer(handle(), buffer);
		glNamedFramebufferDrawBuffers(handle(), 1, &buffer);
	}

	void Framebuffer::addTextureAttachment(GLenum attachment, Texture2D* texture, int level)
	{
		glNamedFramebufferTexture(m_Handle, attachment, texture->handle(), level);
	}

	void Framebuffer::detachTextureAttachment(GLenum attachment)
	{
		glNamedFramebufferTexture(m_Handle, attachment, NULL, 0);
	}

	void Framebuffer::ensureComplete()
	{
		const GLint status = glCheckNamedFramebufferStatus(m_Handle, GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) throw UTAD_EXCEPTION(String("Framebuffer is not complete: ").append(std::to_string(status)));
	}

	void Framebuffer::bindDefault()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}