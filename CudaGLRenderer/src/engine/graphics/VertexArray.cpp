#include "engine/graphics/VertexArray.h"

namespace utad
{
	uint VertexAttrib::size() const
	{
		uint dataSize = 1;

		switch (dataSize)
		{
		case GL_FLOAT: 
			dataSize = sizeof(float);
			break;
		case GL_INT:
		case GL_UNSIGNED_INT:
			dataSize = sizeof(int);
			break;
		case GL_BYTE:
		case GL_UNSIGNED_BYTE:
			dataSize = sizeof(byte);
			break;
		case GL_DOUBLE:
			dataSize = sizeof(double);
			break;
		}

		return count * dataSize;
	}

	uint VertexAttribList::stride() const
	{
		uint stride = 0;
		for (const VertexAttrib& attrib : attributes)
		{
			stride += attrib.size();
		}
		return stride;
	}

	VertexArray::VertexArray()
	{
		glGenVertexArrays(1, &m_Handle);
	}

	VertexArray::~VertexArray()
	{
		if (m_DestroyBuffersOnDelete)
		{
			for (auto [binding, buffer] : m_VertexBuffers)
			{
				UTAD_DELETE(buffer);
			}
			UTAD_DELETE(m_IndexBuffer);
		}

		glDeleteVertexArrays(1, &m_Handle);
	}

	Handle VertexArray::handle() const
	{
		return m_Handle;
	}

	VertexBuffer* VertexArray::vertexBuffer(uint binding) const
	{
		if (m_VertexBuffers.find(binding) == m_VertexBuffers.end()) return nullptr;
		return m_VertexBuffers.at(binding);
	}

	bool VertexArray::addVertexBuffer(uint binding, VertexBuffer* buffer, uint stride)
	{
		if (buffer == nullptr || vertexBuffer(binding) == buffer) return false;
		m_VertexBuffers[binding] = buffer;
		glVertexArrayVertexBuffer(m_Handle, binding, buffer->handle(), 0, stride);
		return true;
	}

	bool VertexArray::addVertexBuffer(uint binding, VertexBuffer* buffer, const VertexAttribList& attributes)
	{
		if(!addVertexBuffer(binding, buffer, attributes.stride())) return false;
		setVertexAttribs(binding, attributes);
		return true;
	}

	void VertexArray::setVertexAttribs(uint binding, const VertexAttribList& attributes)
	{
		uint location = 0;
		uint offset = 0;
		for (const VertexAttrib& attrib : attributes.attributes)
		{
			setVertexAttrib(binding, attrib, location, offset);
			++location;
			offset += attrib.size();
		}
	}

	void VertexArray::setVertexAttrib(uint binding, const VertexAttrib& attrib, uint location, uint offset)
	{
		glEnableVertexArrayAttrib(m_Handle, location);
		glVertexArrayAttribBinding(m_Handle, location, binding);

		if (attrib.type == GL_FLOAT || attrib.type == GL_DOUBLE)
			glVertexArrayAttribFormat(handle(), location, attrib.count, attrib.type, false, offset);
		else
			glVertexArrayAttribIFormat(handle(), location, attrib.count, attrib.type, offset);
	}

	IndexBuffer* VertexArray::indexBuffer() const
	{
		return m_IndexBuffer;
	}

	void VertexArray::setIndexBuffer(IndexBuffer* buffer)
	{
		if(buffer == nullptr) return;
		m_IndexBuffer = buffer;
		glVertexArrayElementBuffer(m_Handle, buffer->handle());
	}

	void VertexArray::bind()
	{
		glBindVertexArray(m_Handle);
	}

	void VertexArray::unbind()
	{
		glBindVertexArray(0);
	}

	void VertexArray::destroyVertexBuffers()
	{
		for (auto& pair : m_VertexBuffers)
		{
			UTAD_DELETE(pair.second);
		}
		m_VertexBuffers.clear();
	}

	void VertexArray::destroyIndexBuffer()
	{
		UTAD_DELETE(m_IndexBuffer);
	}

	void VertexArray::setDestroyBuffersOnDelete(bool destroyBuffers)
	{
		m_DestroyBuffersOnDelete = destroyBuffers;
	}
}