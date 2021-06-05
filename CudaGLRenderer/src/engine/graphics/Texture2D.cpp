#include "engine/graphics/Texture.h"
#include "engine/Common.h"

namespace utad
{
	Texture2D::Texture2D() : m_Width(0), m_Height(0)
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &m_Handle);
	}

	Texture2D::~Texture2D()
	{
		glDeleteTextures(1, &m_Handle);
		m_Handle = NULL;
	}

	void Texture2D::allocate(const TextureAllocInfo& allocInfo)
	{
		int levels = allocInfo.levels == 0 ? Texture::mipLevelsOf(allocInfo.width, allocInfo.height) : allocInfo.levels;
		glTextureStorage2D(m_Handle, levels, allocInfo.format, allocInfo.width, allocInfo.height);
		m_Width = allocInfo.width;
		m_Height = allocInfo.height;
	}

	void Texture2D::update(const Texture2DUpdateInfo& updateInfo)
	{
		glTextureSubImage2D(m_Handle, 0, 0, 0, m_Width, m_Height, updateInfo.format, updateInfo.type, updateInfo.pixels);
	}

	void Texture2D::filter(GLenum filterType, GLenum filter)
	{
		glTextureParameteri(m_Handle, filterType, filter);
	}

	void Texture2D::wrap(GLenum mode)
	{
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_R, mode);
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_S, mode);
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_T, mode);
	}

	void Texture2D::wrap(GLenum coord, GLenum clamp)
	{
		glTextureParameteri(m_Handle, coord, clamp);
	}

	void Texture2D::generateMipmaps()
	{
		glGenerateTextureMipmap(m_Handle);
	}

	int Texture2D::width() const
	{
		return m_Width;
	}

	int Texture2D::height() const
	{
		return m_Height;
	}

	void Texture2D::bind(int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_2D, m_Handle);
	}

	void Texture2D::unbind(int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	int Texture::mipLevelsOf(int width, int height)
	{
		return (int) math::log2(math::max((float)width, (float)height) + 1.0f);
	}
}