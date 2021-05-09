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
	public:
		VertexArray();
		~VertexArray();
		Handle handle() const;
		void addVertexBuffer(uint binding, VertexBuffer* buffer, const VertexAttribList& attributes);
		void setIndexBuffer(IndexBuffer* buffer);
		void bind();
		void unbind();
	private:
		void setVertexAttrib(uint binding, const VertexAttrib& attrib, uint location, uint offset);
	};
}