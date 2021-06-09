#include "engine/assets/Mesh.h"

namespace utad
{
	Mesh::Mesh(VertexArray* vertexArray) : m_VertexArray(vertexArray)
	{
	}

	Mesh::~Mesh()
	{
		UTAD_DELETE(m_VertexArray);
	}

	VertexArray* Mesh::vertexArray() const
	{
		return m_VertexArray;
	}

	GLenum Mesh::drawMode() const
	{
		return m_DrawMode;
	}

	uint Mesh::vertexCount() const
	{
		return m_Vertices.size();
	}

	uint Mesh::indexCount() const
	{
		return m_IndexCount;
	}

	GLenum Mesh::indexType() const
	{
		return m_IndexType;
	}

	uint Mesh::indexBufferOffset() const
	{
		return m_IndexBufferOffset;
	}

	const ArrayList<Vertex>& Mesh::vertices() const
	{
		return m_Vertices;
	}

	void Mesh::bind()
	{
		m_VertexArray->bind();
	}

	void Mesh::unbind()
	{
		m_VertexArray->unbind();
	}
}