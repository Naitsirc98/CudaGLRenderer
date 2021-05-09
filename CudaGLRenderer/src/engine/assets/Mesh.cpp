#include "engine/assets/Mesh.h"

namespace utad
{
	Model::Model() : m_Root(nullptr)
	{
	}

	Model::~Model()
	{
		m_Meshes.clear();
		UTAD_DELETE(m_Root);
	}

	Model::Node* Model::root() const
	{
		return m_Root;
	}

	const ArrayList<Mesh*>& Model::meshes() const
	{
		return m_Meshes;
	}
	
	Model::Node::~Node()
	{
		for (Node* child : children)
		{
			UTAD_DELETE(child);
		}
	}
}