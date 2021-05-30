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
	private:
		Mesh(VertexArray* vao) : m_VertexArray(vao) {}
	public:
		~Mesh() { UTAD_DELETE(m_VertexArray);}
		VertexArray* vertexArray() const { return m_VertexArray; }
	};
}