#include "engine/graphics/VertexArray.h"
#include <algorithm>

namespace utad
{
	VertexArray::VertexArray()
	{
		glCreateVertexArrays(1, &m_Handle);
	}

	VertexArray::~VertexArray()
	{
		if (m_DestroyBuffersOnDelete)
		{
			for (auto [binding, bufferDesc] : m_VertexBuffers)
			{
				UTAD_DELETE(bufferDesc.buffer);
			}
			UTAD_DELETE(m_IndexBuffer);
		}

		glDeleteVertexArrays(1, &m_Handle);
	}

	Handle VertexArray::handle() const
	{
		return m_Handle;
	}

	const VertexBufferDescriptor* VertexArray::get(uint binding) const
	{
		if (m_VertexBuffers.find(binding) == m_VertexBuffers.end()) return nullptr;
		return &m_VertexBuffers.at(binding);
	}

	bool VertexArray::addVertexBuffer(uint binding, VertexBuffer* buffer, uint stride, uint offset)
	{
		if (buffer == nullptr) return false;
		if (get(binding) != nullptr) return false;

		glVertexArrayVertexBuffer(handle(), binding, buffer->handle(), offset, stride);

		VertexBufferDescriptor desc = {};
		desc.binding = binding;
		desc.buffer = buffer;
		desc.offset = offset;
		desc.stride = stride;

		m_VertexBuffers[binding] = std::move(desc);

		return true;
	}

	void VertexArray::removeVertexBuffer(uint binding)
	{
		glVertexArrayVertexBuffer(handle(), 0, NULL, 0, 0);
		m_VertexBuffers.erase(m_VertexBuffers.find(binding));
	}

	void VertexArray::setVertexAttributes(uint binding, const ArrayList<VertexAttrib>& attributes)
	{
		uint offset = 0;
		for (const VertexAttrib& attribute : attributes)
		{
			setVertexAttribute(binding, attribute, offset);
			offset += VertexAttribDescription::of(attribute).size();
		}
	}

	void VertexArray::setVertexAttribute(uint binding, VertexAttrib attrib, uint offset)
	{
		if (attrib == VertexAttrib::None) return;

		const VertexAttribDescription& desc = VertexAttribDescription::of(attrib);

		glEnableVertexArrayAttrib(m_Handle, desc.location);
		glVertexArrayAttribBinding(m_Handle, desc.location, binding);

		if (desc.type == GL_FLOAT || desc.type == GL_DOUBLE)
			glVertexArrayAttribFormat(handle(), desc.location, desc.count, desc.type, false, offset);
		else
			glVertexArrayAttribIFormat(handle(), desc.location, desc.count, desc.type, offset);
	}

	IndexBuffer* VertexArray::indexBuffer() const
	{
		return m_IndexBuffer;
	}

	void VertexArray::setIndexBuffer(IndexBuffer* buffer)
	{
		glVertexArrayElementBuffer(handle(), buffer == nullptr ? NULL : buffer->handle());
		m_IndexBuffer = buffer;
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
		for (auto [binding, bufferDesc] : m_VertexBuffers)
		{
			UTAD_DELETE(bufferDesc.buffer);
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