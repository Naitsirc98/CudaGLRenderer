#include "engine/assets/Image.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include "engine/io/Files.h"

namespace utad
{

	Image::Image(uint32_t width, uint32_t height, GLenum format, void* pixels, ImageDeallocator deallocator_)
		: m_Width(width), m_Height(height), m_Format(format), m_Pixels(pixels), m_Deallocator(deallocator_)
	{
	}

	Image::Image(Image&& move)
	{
		m_Width = move.m_Width;
		m_Height = move.m_Height;
		m_Format = move.format();
		m_Pixels = move.m_Pixels;
		move.m_Pixels = nullptr;
	}

	Image::~Image()
	{
		if (m_Pixels == nullptr) return;
		m_Deallocator(m_Pixels);
		m_Pixels = nullptr;
	}

	Image& Image::operator=(Image&& move)
	{
		if (&move == this) return *this;

		m_Width = move.m_Width;
		m_Height = move.m_Height;
		m_Format = move.m_Format;

		m_Pixels = move.m_Pixels;
		m_Deallocator = move.m_Deallocator;
		move.m_Pixels = nullptr;

		return *this;
	}

	inline void* allocate(GLenum format, uint32_t width, uint32_t height, uint32_t value)
	{
		size_t size = width * height * sizeOfFormat(format);
		void* pixels = malloc(size);
		memset(pixels, value, size);
		return pixels;
	}

	Image* ImageFactory::createWhiteImage(GLenum format, uint32_t width, uint32_t height)
	{
		return createImage(format, width, height, 0xFFFFFFFF);
	}

	Image* ImageFactory::createBlackImage(GLenum format, uint32_t width, uint32_t height)
	{
		return createImage(format, width, height, 0);
	}

	Image* ImageFactory::createImage(GLenum format, uint32_t width, uint32_t height, uint32_t value)
	{
		return createImage(allocate(format, width, height, value), format, width, height);
	}

	Image* ImageFactory::createImage(void* pixels, GLenum format, uint32_t width, uint32_t height)
	{
		return new Image(width, height, format, pixels, free);
	}

	Image* ImageFactory::createImage(const String& path, GLenum format, bool flipY)
	{
		int32_t width;
		int32_t height;
		int32_t channels;
		int32_t desiredChannels = format == format == GL_NONE ? STBI_default : channelsOfFormat(format);
		void* pixels;

		stbi_set_flip_vertically_on_load(flipY);

		if (format != format == GL_NONE && isFloatFormat(format))
			pixels = stbi_loadf(path.c_str(), &width, &height, &channels, desiredChannels);
		else
			pixels = stbi_load(path.c_str(), &width, &height, &channels, desiredChannels);

		stbi_set_flip_vertically_on_load(false);

		if (pixels == nullptr)
			throw UTAD_EXCEPTION(String("Failed to create image from file: ").append(stbi_failure_reason()));

		if (format == GL_NONE) format = formatFromChannels(channels);

		return new Image(width, height, format, pixels, stbi_image_free);
	}

	uint32_t sizeOfFormat(GLenum format)
	{
		uint32_t channels = channelsOfFormat(format);
		return isFloatFormat(format) ? channels * sizeof(float) : channels;
	}

	uint32_t bitsOfFormat(GLenum format)
	{
		uint32_t channels = channelsOfFormat(format);
		return isFloatFormat(format) ? 16 : 8;
	}

	uint32_t channelsOfFormat(GLenum format)
	{
		switch (format)
		{
		case GL_RED:
		case GL_R16F:
		case GL_R32F:
			return 1;
		case GL_RG:
		case GL_RG16F:
		case GL_RG32F:
			return 2;
		case GL_RGB:
		case GL_RGB16F:
		case GL_RGB32F:
			return 3;
		case GL_RGBA:
		case GL_RGBA16F:
		case GL_RGBA32F:
			return 4;
		}
		return -1;
	}

	bool isFloatFormat(GLenum format)
	{
		switch (format)
		{
		case GL_RGB16F:
		case GL_RGB32F:
		case GL_RGBA16F:
		case GL_RGBA32F:
			return true;
		}
		return false;
	}

	GLenum formatFromChannels(uint32_t channels, bool floatFormat)
	{
		switch (channels)
		{
		case 1:
			return floatFormat ? GL_R16F : GL_RED;
		case 2:
			return floatFormat ? GL_RG16F : GL_RG;
		case 3:
			return floatFormat ? GL_RGB16F : GL_RGB;
		case 4:
			return floatFormat ? GL_RGBA16F : GL_RGBA;
		}
		return GL_NONE;
	}

	GLenum getPixelFormatFrom(int component)
	{
		switch (component)
		{
			case 1: return GL_RED;
			case 2: return GL_RG;
			case 3: return GL_RGB;
			case 4: return GL_RGBA;
		}
		throw UTAD_EXCEPTION(String("Unknown format for components: ").append(std::to_string(component)));
	}

	GLenum toSizedFormatRED(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_R8;
			case 16: return type == GL_FLOAT ? GL_R16F : GL_R16;
			case 32: return type == GL_FLOAT ? GL_R32F : GL_R32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	GLenum toSizedFormatRG(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RG8;
			case 16: return type == GL_FLOAT ? GL_RG16F : GL_RG16;
			case 32: return type == GL_FLOAT ? GL_RG32F : GL_RG32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	GLenum toSizedFormatRGB(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RGB8;
			case 16: return type == GL_FLOAT ? GL_RGB16F : GL_RGB16;
			case 32: return type == GL_FLOAT ? GL_RGB32F : GL_RGB32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	GLenum toSizedFormatRGBA(int bits, int type)
	{
		switch (bits)
		{
			case 8: return GL_RGBA8;
			case 16: return type == GL_FLOAT ? GL_RGBA16F : GL_RGBA16;
			case 32: return type == GL_FLOAT ? GL_RGBA32F : GL_RGBA32UI;
		}
		throw UTAD_EXCEPTION(String("Unknown combination of bits and type: ").append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}

	GLenum toSizedFormat(GLenum format, int bits, int type)
	{
		switch (format)
		{
			case GL_RED: return toSizedFormatRED(bits, type);
			case GL_RG: return toSizedFormatRG(bits, type);
			case GL_RGB: return toSizedFormatRGB(bits, type);
			case GL_RGBA: return toSizedFormatRGBA(bits, type);
		}
		throw UTAD_EXCEPTION(String("Unknown combination of format, bits and type: ").append(std::to_string(format)).append(" + ")
			.append(std::to_string(bits)).append(" + ").append(std::to_string(type)));
	}
}