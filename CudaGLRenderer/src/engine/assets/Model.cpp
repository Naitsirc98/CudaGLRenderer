#include "engine/assets/Model.h"

namespace utad
{
	Model::Model()
	{
	}

	Model::~Model()
	{
		for (ModelNode* node : m_Nodes)
		{
			UTAD_DELETE(node);
		}
		m_Meshes.clear();
	}

	const ArrayList<ModelNode*>& Model::nodes() const
	{
		return m_Nodes;
	}

	const ArrayList<Mesh*>& Model::meshes() const
	{
		return m_Meshes;
	}

	const ArrayList<Material*>& Model::materials() const
	{
		return m_Materials;
	}

	ModelNode* Model::createNode(ModelNode* parent)
	{
		ModelNode* node = new ModelNode(*this, m_Nodes.size());
		m_Nodes.push_back(node);
		if (parent != nullptr)
		{
			node->m_Parent = parent;
			parent->m_Children.push_back(node);
		}
		return node;
	}

	ModelNode::ModelNode(Model& model, uint index) : m_Model(model), m_Index(index)
	{
	}
	
	ModelNode::~ModelNode()
	{
		for (ModelNode* child : m_Children)
		{
			UTAD_DELETE(child);
		}
	}

	uint ModelNode::index() const
	{
		return m_Index;
	}

	const String& ModelNode::name() const
	{
		return m_Name;
	}

	uint ModelNode::mesh() const
	{
		return m_Mesh;
	}

	uint ModelNode::material() const
	{
		return m_Material;
	}

	const Matrix4& ModelNode::transformation() const
	{
		return m_Transformation;
	}

	const ArrayList<ModelNode*>& ModelNode::children() const
	{
		return m_Children;
	}

	const Model& ModelNode::model() const
	{
		return m_Model;
	}

	const ModelNode* ModelNode::parent() const
	{
		return m_Parent;
	}

	ModelNode* ModelNode::createChild()
	{
		return m_Model.createNode(this);
	}


}