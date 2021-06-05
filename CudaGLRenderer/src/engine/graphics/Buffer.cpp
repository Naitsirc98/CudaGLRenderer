#include "engine/graphics/Buffer.h"

namespace utad
{
	Buffer::Buffer()
	{
		glCreateBuffers(1, &m_Handle);
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

	void Buffer::bind(GLenum target)
	{
		glBindBuffer(target, m_Handle);
	}

	void Buffer::unbind(GLenum target)
	{
		glBindBuffer(target, m_Handle);
	}

	void Buffer::allocate(const BufferAllocInfo& allocInfo)
	{
		glNamedBufferStorage(m_Handle, allocInfo.size, allocInfo.data, allocInfo.storageFlags);
		m_Size = allocInfo.size;
	}

	void Buffer::update(const BufferUpdateInfo& updateInfo)
	{
		glNamedBufferSubData(m_Handle, updateInfo.offset, updateInfo.size, updateInfo.data);
	}

	byte* Buffer::mapMemory(GLenum mapFlags)
	{
		if (m_MappedMemory != nullptr) return m_MappedMemory;
		m_MappedMemory = static_cast<byte*>(glMapNamedBuffer(m_Handle, mapFlags));
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

	size_t Buffer::size() const
	{
		return m_Size;
	}
}