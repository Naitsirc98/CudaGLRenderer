#pragma once

#include "engine/graphics/VertexArray.h"

namespace utad
{
	using MeshID = uint;

	class Mesh
	{
		friend class AssetsManager;
		friend class ModelLoader;
	private:
		VertexArray* m_VertexArray{nullptr};
		GLenum m_DrawMode;
		uint m_IndexCount;
		GLenum m_IndexType;
		uint m_IndexBufferOffset;
	private:
		Mesh(VertexArray* vao) : m_VertexArray(vao) {}
	public:
		~Mesh() { UTAD_DELETE(m_VertexArray);}
		VertexArray* vertexArray() const { return m_VertexArray; }
		GLenum drawMode() const {return m_DrawMode;};
		uint indexCount() const { return m_IndexCount; };
		GLenum indexType() const {return m_IndexType;};
		uint indexBufferOffset() const { return m_IndexBufferOffset; };
	};
}