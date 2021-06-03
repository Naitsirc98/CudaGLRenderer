#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	const GLenum GPU_STORAGE_LOCAL_FLAGS = 0;
	const GLenum CPU_STORAGE_WRITE_FLAGS = GL_DYNAMIC_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT;
	const GLenum CPU_STORAGE_READ_WRITE_FLAGS = GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT;


	struct BufferAllocInfo
	{
		size_t size{0};
		void* data{nullptr};
		GLenum storageFlags{GPU_STORAGE_LOCAL_FLAGS};
	};

	struct BufferUpdateInfo
	{
		size_t offset{0};
		size_t size{0};
		void* data{nullptr};
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
		void bind(GLenum target);
		void unbind(GLenum target);
		void allocate(const BufferAllocInfo& allocInfo);
		void update(const BufferUpdateInfo& updateInfo);
		byte* mapMemory(GLenum mapFlags);
		byte* mappedMemory() const;
		void unmapMemory();
	};

	using VertexBuffer = Buffer;
	using IndexBuffer = Buffer;
	using UniformBuffer = Buffer;
	using StorageBuffer = Buffer;

	struct BufferView
	{
		Buffer* buffer{nullptr};
		GLenum target{0};
		size_t offset{0};
		size_t size{0};
	};
}