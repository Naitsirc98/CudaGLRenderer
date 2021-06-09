#include "engine/graphics/VertexArray.h"
#include <algorithm>

namespace utad
{
	static const VertexAttribDescription NoneDesc      = {-1, GL_FLOAT, 0};
	static const VertexAttribDescription PositionDesc  = {0, GL_FLOAT, 3};
	static const VertexAttribDescription NormalDesc    = {1, GL_FLOAT, 3};
	static const VertexAttribDescription TexCoordsDesc = {2, GL_FLOAT, 2};
	static const VertexAttribDescription TangentsDesc  = {3, GL_FLOAT, 3};
	static const VertexAttribDescription ColorDesc     = {4, GL_FLOAT, 3};

	static const VertexAttribDescription StandardAttribs[] {
		PositionDesc, NormalDesc, TexCoordsDesc, TangentsDesc, ColorDesc
	};

	static const String StandardAttribsNames[] {
		"Position", "Normal", "TexCoords", "Tangents", "Color"
	};

	const VertexAttribDescription& VertexAttribDescription::of(VertexAttrib attribute)
	{
		return StandardAttribs[static_cast<int>(attribute) - 1];
	}

	const String& VertexAttribDescription::name(VertexAttrib attribute)
	{
		return StandardAttribsNames[static_cast<int>(attribute) - 1];
	}

	uint VertexAttribDescription::size() const
	{
		int dataSize = 1;

		switch (type)
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