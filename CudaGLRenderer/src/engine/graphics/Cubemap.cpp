#include "engine/graphics/Texture.h"

namespace utad
{
	Cubemap::Cubemap()
	{
		glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &m_Handle);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	}

	Cubemap::~Cubemap()
	{
		unmapPixelsAllFaces();
		glDeleteTextures(1, &m_Handle);
	}

	void Cubemap::allocate(const TextureAllocInfo& allocInfo)
	{
		int levels = allocInfo.levels == 0 ? Texture::mipLevelsOf(allocInfo.width, allocInfo.height) : allocInfo.levels;
		glTextureStorage2D(m_Handle, levels, allocInfo.format, allocInfo.width, allocInfo.height);
		m_Width = allocInfo.width;
		m_Height = allocInfo.height;
	}

	void Cubemap::update(const CubemapUpdateInfo& updateInfo)
	{
		bind();

		const Vector2& offset = updateInfo.offset;
		const GLenum format = updateInfo.format;
		const GLenum type = updateInfo.type;
		const GLenum face = updateInfo.face;

		glTexSubImage2D(face, updateInfo.level, offset.x, offset.y, m_Width, m_Height, format, type, updateInfo.pixels);

		unbind();
	}

	void Cubemap::filter(GLenum filterType, GLenum filter)
	{
		glTextureParameteri(m_Handle, filterType, filter);
	}

	void Cubemap::wrap(GLenum mode)
	{
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_R, mode);
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_S, mode);
		glTextureParameteri(m_Handle, GL_TEXTURE_WRAP_T, mode);
	}

	void Cubemap::wrap(GLenum coord, GLenum clamp)
	{
		glTextureParameteri(m_Handle, coord, clamp);
	}

	void Cubemap::generateMipmaps()
	{
		glGenerateTextureMipmap(m_Handle);
	}

	int Cubemap::width() const
	{
		return m_Width;
	}

	int Cubemap::height() const
	{
		return m_Height;
	}

	void Cubemap::bind(int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_Handle);
	}

	void Cubemap::unbind(int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	}

	byte* Cubemap::pixels(GLenum face) const
	{
		return m_Pixels[face];
	}

	byte* Cubemap::mapPixels(GLenum face, GLenum format, GLenum type, size_t size)
	{
		if (m_Pixels[face] != nullptr) unmapPixels(face);

		m_Pixels[face] = new byte[size];

		glBindTexture(face, m_Handle);
		glGetTexImage(face, 0, format, type, m_Pixels[face]);
		glBindTexture(face, m_Handle);

		return m_Pixels[face];
	}

	void Cubemap::unmapPixels(GLenum face)
	{
		UTAD_DELETE(m_Pixels[face]);
	}

	void Cubemap::unmapPixelsAllFaces()
	{
		for (size_t i = 0; i < 6; ++i)
			unmapPixels(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);
	}

}