#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	struct BufferAllocInfo
	{
		size_t size;
		void* data;
	};

	struct BufferUpdateInfo
	{
		size_t offset;
		size_t size;
		void* data;
	};

	class Buffer
	{
	private:
		Handle m_Handle{NULL};
		size_t m_Size{0};
		byte* m_MappedMemory{nullptr};
	public:
		Buffer();
		virtual ~Buffer();
		Handle handle() const;
		void bind();
		void unbind();
		void allocate(const BufferAllocInfo& allocInfo);
		void update(const BufferUpdateInfo& updateInfo);
		byte* mapMemory();
		byte* mappedMemory() const;
		void unmapMemory();
		virtual GLenum type() const = 0;
		virtual GLint storageFlags() const = 0;
		virtual GLint mapMemoryFlags() const = 0;
	};

	class VertexBuffer : public Buffer
	{
	public:
		GLenum type() const override;
		GLint storageFlags() const override;
		GLint mapMemoryFlags() const override;
	};

	class IndexBuffer : public Buffer
	{
	public:
		GLenum type() const override;
		GLint storageFlags() const override;
		GLint mapMemoryFlags() const override;
	};

	class UniformBuffer : public Buffer
	{
	public:
		GLenum type() const override;
		GLint storageFlags() const override;
		GLint mapMemoryFlags() const override;
	};

	class StorageBuffer : public Buffer
	{
	public:
		GLenum type() const override;
		GLint storageFlags() const override;
		GLint mapMemoryFlags() const override;
	};
}