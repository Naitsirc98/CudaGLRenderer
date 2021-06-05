#pragma once

#include "engine/graphics/GraphicsAPI.h"

namespace utad
{
	using ImageDeallocator = Function<void, void*>;

	uint32_t sizeOfFormat(GLenum format);
	uint32_t bitsOfFormat(GLenum format);
	uint32_t channelsOfFormat(GLenum format);
	bool isFloatFormat(GLenum format);
	GLenum formatFromChannels(uint32_t channels, bool floatFormat = false);
	GLenum getPixelFormatFrom(int component);
	GLenum toSizedFormatRED(int bits, int type);
	GLenum toSizedFormatRG(int bits, int type);
	GLenum toSizedFormatRGB(int bits, int type);
	GLenum toSizedFormatRGBA(int bits, int type);
	GLenum toSizedFormat(GLenum format, int bits, int type);

	class Image
	{
	private:
		uint32_t m_Width;
		uint32_t m_Height;
		GLenum m_Format;
		void* m_Pixels;
	private:
		ImageDeallocator m_Deallocator;
	public:
		Image(uint32_t width, uint32_t height, GLenum format, void* pixels, ImageDeallocator deallocator_ = free);
		Image(const Image& copy) = delete;
		Image(Image&& move);
		~Image();
		Image& operator=(const Image& copy) = delete;
		Image& operator=(Image&& move);
		size_t size() const { return m_Width * m_Height * sizeOfFormat(m_Format); }
		uint32_t width() const { return m_Width; }
		uint32_t height() const { return m_Height; }
		GLenum format() const { return m_Format; }
		GLenum type() const { return isFloatFormat(m_Format) ? GL_FLOAT : GL_UNSIGNED_BYTE; }
		const void* pixels() const { return m_Pixels; }
		void* pixels() { return m_Pixels; }
	};

	class ImageFactory
	{
	public:
		static Image * createWhiteImage(GLenum format, uint32_t width = 1, uint32_t height = 1);
		static Image* createBlackImage(GLenum format, uint32_t width = 1, uint32_t height = 1);
		static Image* createImage(const String& path, GLenum format, bool flipY);
		static Image* createImage(void* pixels, GLenum format, uint32_t width = 1, uint32_t height = 1);
		static Image* createImage(GLenum format, uint32_t width = 1, uint32_t height = 1,  uint32_t value = 0);
	private:
		ImageFactory() = delete;
	};

}