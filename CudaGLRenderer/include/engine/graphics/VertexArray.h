#pragma once

#include "Vertex.h"
#include "Buffer.h"

namespace utad
{
	struct VertexBufferDescriptor
	{
		uint binding;
		VertexBuffer* buffer{ nullptr };
		ArrayList<VertexAttrib> attributes;
		uint offset;
		uint stride;
	};

	class VertexArray
	{
	private:
		Handle m_Handle;
		Map<uint, VertexBufferDescriptor> m_VertexBuffers;
		IndexBuffer* m_IndexBuffer{nullptr};
		bool m_DestroyBuffersOnDelete{true};
	public:
		VertexArray();
		~VertexArray();
		Handle handle() const;
		const VertexBufferDescriptor* get(uint binding) const;
		bool addVertexBuffer(uint binding, VertexBuffer* buffer, uint stride, uint offset = 0);
		void removeVertexBuffer(uint binding);
		void setVertexAttributes(uint binding, const ArrayList<VertexAttrib>& attributes);
		void setVertexAttribute(uint binding, VertexAttrib attribute, uint offset);
		IndexBuffer* indexBuffer() const;
		void setIndexBuffer(IndexBuffer* buffer);
		void bind();
		void unbind();
		void destroyVertexBuffers();
		void destroyIndexBuffer();
		void setDestroyBuffersOnDelete(bool destroyBuffers = true);
	public:
		template<typename Container>
		static int stride(const Container& container)
		{
			int stride = 0;
			for (const VertexAttrib& attrib : attributes)
			{
				stride += VertexAttribDescription::of(attrib).size();
			}
			return stride;
		}
	};
}