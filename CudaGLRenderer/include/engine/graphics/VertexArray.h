#pragma once

#include "Buffer.h"

namespace utad
{
	struct VertexAttrib
	{
		GLenum type;
		GLsizei count;
		uint size() const;
	};

	struct VertexAttribList
	{
		ArrayList<VertexAttrib> attributes;
		uint stride() const;
	};

	class VertexArray
	{
	private:
		Handle m_Handle;
		Map<uint, VertexBuffer*> m_VertexBuffers;
		IndexBuffer* m_IndexBuffer;
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