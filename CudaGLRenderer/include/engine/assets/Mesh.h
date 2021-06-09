#pragma once

#include "engine/graphics/VertexArray.h"

namespace utad
{
	using MeshID = uint;

	class Mesh
	{
		friend class AssetsManager;
		friend class ModelLoader;
		friend class MeshPrimitives;
	private:
		VertexArray* m_VertexArray{nullptr};
		GLenum m_DrawMode{0};
		uint m_IndexCount{0};
		GLenum m_IndexType{0};
		uint m_IndexBufferOffset{0};
		ArrayList<Vertex> m_Vertices;
	public:
		Mesh(VertexArray* vao);
		~Mesh();
		VertexArray* vertexArray() const;
		GLenum drawMode() const;
		uint vertexCount() const;
		uint indexCount() const;
		GLenum indexType() const;
		uint indexBufferOffset() const;
		const ArrayList<Vertex>& vertices() const;
		void bind();
		void unbind();
	};
}