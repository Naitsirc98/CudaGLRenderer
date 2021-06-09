#pragma once

#include "Buffer.h"

namespace utad
{
	struct Vertex
	{
		Vector4 position;
		Vector3 normal;
		Vector2 texCoords;
		Vector3 tangent;
		Color color;
	};

	enum class VertexAttrib
	{
		None,
		Position,
		Normal,
		TexCoords,
		Tangent,
		Color
	};

	struct VertexAttribDescription
	{
		GLuint location;
		GLenum type;
		GLsizei count;
		uint size() const;

		static const VertexAttribDescription& of(VertexAttrib attribute);
		static const String& name(VertexAttrib attribute);
	};

	struct VertexBufferDescriptor
	{
		uint binding;
		VertexBuffer* buffer{nullptr};
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