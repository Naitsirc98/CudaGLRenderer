#pragma once

#include "engine/graphics/VertexArray.h"

namespace utad
{
	using MeshID = uint;

	class Mesh
	{
		friend class AssetsManager;
	private:
		MeshID m_ID{NULL};
		VertexArray* m_VertexArray{nullptr};
	private:
		Mesh(MeshID id) : m_ID(id), m_VertexArray(nullptr) {}
		~Mesh() { UTAD_DELETE(m_VertexArray); m_ID = NULL; }
		void setVertexArray(VertexArray* vao) { m_VertexArray = vao; }
	public:
		MeshID id() const { return m_ID; }
		VertexArray* vertexArray() const { return m_VertexArray; }
	};
}