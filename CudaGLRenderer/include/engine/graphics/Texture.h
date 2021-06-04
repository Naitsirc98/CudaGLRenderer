#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	struct TextureAllocInfo
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

	struct CubemapUpdateInfo
	{
		GLenum face;
		Vector2 offset;
		int level;
		GLenum format;
		GLenum type;
		void* pixels;
	};

	class Texture
	{
	public:
		static int mipLevelsOf(int width, int height);

		virtual Handle handle() const = 0;
		virtual void bind(int unit = 0) = 0;
		virtual void unbind(int unit = 0) = 0;
	};

	class Texture2D : public Texture
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
		Handle handle() const override { return m_Handle; }
		void allocate(const TextureAllocInfo& allocInfo);
		void update(const Texture2DUpdateInfo& updateInfo);
		void filter(GLenum filterType, GLenum filter);
		void wrap(GLenum clamp);
		void wrap(GLenum coord, GLenum clamp);
		void generateMipmaps();
		int width() const;
		int height() const;
		void bind(int unit = 0) override;
		void unbind(int unit = 0) override;
	};

	class Cubemap : public Texture
	{
	private:
		Handle m_Handle;
		int m_Width;
		int m_Height;
	public:
		Cubemap();
		Cubemap(const Cubemap& other) = delete;
		~Cubemap();
		Cubemap& operator=(const Cubemap& other) = delete;
		Handle handle() const override { return m_Handle; }
		void allocate(const TextureAllocInfo& allocInfo);
		void update(const CubemapUpdateInfo& updateInfo);
		void filter(GLenum filterType, GLenum filter);
		void wrap(GLenum clamp);
		void wrap(GLenum coord, GLenum clamp);
		void generateMipmaps();
		int width() const;
		int height() const;
		void bind(int unit = 0) override;
		void unbind(int unit = 0) override;
	};
}