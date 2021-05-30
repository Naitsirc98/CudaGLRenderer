#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	struct Texture2DAllocInfo
	{
		int width;
		int height;
		int levels;
		GLenum format;
	};

	struct Texture2DUpdateInfo
	{
		int level;
		GLenum format;
		GLenum type;
		void* pixels;
	};

	class Texture2D
	{
	private:
		Handle m_Handle;
		int m_Width;
		int m_Height;
	public:
		Texture2D();
		Texture2D(const Texture2D& other) = delete;
		~Texture2D();
		Texture2D& operator=(const Texture2D& other) = delete;
		Handle handle() const { return m_Handle; }
		void allocate(const Texture2DAllocInfo& allocInfo);
		void update(const Texture2DUpdateInfo& updateInfo);
		void filter(GLenum minFilter, GLenum maxFilter);
		void wrap(GLenum clamp);
		void wrap(GLenum coord, GLenum clamp);
		void generateMipmaps();
		int width() const;
		int height() const;
		void bind(int unit = 0);
	public:
		static int mipLevelsOf(int width, int height);
	};
}