#include "engine/graphics/Buffer.h"

namespace utad
{
	Buffer* Buffer::create(GLenum target)
	{
		switch (target)
		{
		case GL_ARRAY_BUFFER: return new VertexBuffer();
		case GL_ELEMENT_ARRAY_BUFFER: return new IndexBuffer();
		case GL_UNIFORM_BUFFER: return new UniformBuffer();
		case GL_SHADER_STORAGE_BUFFER: return new StorageBuffer();
		}
		
		throw UTAD_EXCEPTION(String("Unknown buffer target ").append(std::to_string(target)));
	}


	Buffer::Buffer()
	{
		glGenBuffers(1, &m_Handle);
	}

	Buffer::~Buffer()
	{
		glDeleteBuffers(1, &m_Handle);
		m_Handle = NULL;
	}

	Handle Buffer::handle() const
	{
		return m_Handle;
	}

	void Buffer::bind()
	{
		glBindBuffer(type(), m_Handle);
	}

	void Buffer::unbind()
	{
		glBindBuffer(type(), m_Handle);
	}

	void Buffer::allocate(const BufferAllocInfo& allocInfo)
	{
		glNamedBufferStorage(m_Handle, allocInfo.size, allocInfo.data, storageFlags());
	}

	void Buffer::update(const BufferUpdateInfo& updateInfo)
	{
		glNamedBufferSubData(m_Handle, updateInfo.offset, updateInfo.size, updateInfo.data);
	}

	byte* Buffer::mapMemory()
	{
		if (m_MappedMemory != nullptr) return m_MappedMemory;
		m_MappedMemory = static_cast<byte*>(glMapNamedBuffer(m_Handle, mapMemoryFlags()));
		return m_MappedMemory;
	}

	byte* Buffer::mappedMemory() const
	{
		return m_MappedMemory;
	}

	void Buffer::unmapMemory()
	{
		glUnmapNamedBuffer(m_Handle);
		m_MappedMemory = nullptr;
	}

	//  VERTEX BUFFER

	GLenum VertexBuffer::type() const
	{
		return GL_ARRAY_BUFFER;
	}

	GLint VertexBuffer::storageFlags() const
	{
		return GL_MAP_READ_BIT;
	}

	GLint VertexBuffer::mapMemoryFlags() const
	{
		return GL_READ_ONLY;
	}

	// INDEX BUFFER

	GLenum IndexBuffer::type() const
	{
		return GL_ELEMENT_ARRAY_BUFFER;
	}

	GLint IndexBuffer::storageFlags() const
	{
		return GL_MAP_READ_BIT;
	}

	GLint IndexBuffer::mapMemoryFlags() const
	{
		return GL_READ_ONLY;
	}

	// UNIFORM BUFFER

	GLenum UniformBuffer::type() const
	{
		return GL_UNIFORM_BUFFER;
	}

	GLint UniformBuffer::storageFlags() const
	{
		return GL_DYNAMIC_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT;
	}

	GLint UniformBuffer::mapMemoryFlags() const
	{
		return GL_WRITE_ONLY;
	}

	// STORAGE BUFFER

	GLenum StorageBuffer::type() const
	{
		return GL_SHADER_STORAGE_BUFFER;
	}

	GLint StorageBuffer::storageFlags() const
	{
		return GL_DYNAMIC_STORAGE_BIT | GL_CLIENT_STORAGE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_READ_BIT;
	}

	GLint StorageBuffer::mapMemoryFlags() const
	{
		return GL_READ_WRITE;
	}
}