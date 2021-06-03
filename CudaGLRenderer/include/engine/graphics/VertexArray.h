#pragma once

#include "Buffer.h"

namespace utad
{
	struct VertexAttrib
	{
		GLuint location;
		GLenum type;
		GLsizei count;
		uint size() const;

		bool operator<(const VertexAttrib& other) const
		{
			return location < other.location;
		}

		bool operator>(const VertexAttrib& other) const
		{
			return location > other.location;
		}

		bool operator<=(const VertexAttrib& other) const
		{
			return location <= other.location;
		}

		bool operator>=(const VertexAttrib& other) const
		{
			return location >= other.location;
		}
	};

	class VertexAttribList
	{
	private:
		ArrayList<VertexAttrib> m_Attributes;
	public:
		VertexAttribList(size_t initialCapacity = 3);
		void add(const VertexAttrib& attrib);
		void add(VertexAttrib&& attrib);
		ArrayList<VertexAttrib>::const_iterator begin() const;
		ArrayList<VertexAttrib>::const_iterator end() const;
		uint stride() const;
	};

	class VertexArray
	{
	private:
		Handle m_Handle;
		Map<uint, VertexBuffer*> m_VertexBuffers;
		IndexBuffer* m_IndexBuffer{nullptr};
		bool m_DestroyBuffersOnDelete{true};
	public:
		VertexArray();
		~VertexArray();
		Handle handle() const;
		VertexBuffer* vertexBuffer(uint binding) const;
		bool addVertexBuffer(uint binding, VertexBuffer* buffer, uint stride);
		bool addVertexBuffer(uint binding, VertexBuffer* buffer, const VertexAttribList& attributes);
		void setVertexAttribs(uint binding, const VertexAttribList& attributes);
		void setVertexAttrib(uint binding, const VertexAttrib& attrib, uint location, uint offset);
		IndexBuffer* indexBuffer() const;
		void setIndexBuffer(IndexBuffer* buffer);
		void bind();
		void unbind();
		void destroyVertexBuffers();
		void destroyIndexBuffer();
		void setDestroyBuffersOnDelete(bool destroyBuffers = true);
	};
}