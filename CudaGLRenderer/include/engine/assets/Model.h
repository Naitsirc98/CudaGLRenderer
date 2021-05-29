#pragma once

#include "Mesh.h"
#include "Material.h"

namespace utad
{
	class ModelNode
	{
		friend class Model;
		friend class ModelLoader;
	private:
		Model& m_Model;
		const uint m_Index;
		ModelNode* m_Parent;
		String m_Name;
		ArrayList<uint> m_Meshes;
		ArrayList<uint> m_Materials;
		ArrayList<ModelNode*> m_Children;
	private:
		ModelNode(Model& model, uint index);
		~ModelNode();
	public:
		uint index() const;
		const String& name() const;
		const ArrayList<uint>& meshes() const;
		const ArrayList<uint>& materials() const;
		const ArrayList<ModelNode*>& children() const;
		const Model& model() const;
		const ModelNode* parent() const;
	private:
		ModelNode* createChild();
	};

	class Model
	{
		friend class ModelLoader;
		friend class ModelNode;
	private:
		ArrayList<ModelNode*> m_Nodes;
		ArrayList<Mesh*> m_Meshes;
		ArrayList<Material*> m_Materials;
	public:
		Model();
		~Model();
		const ArrayList<ModelNode*>& nodes() const;
		const ArrayList<Mesh*>& meshes() const;
		const ArrayList<Material*>& materials() const;
	private:
		ModelNode* createNode(ModelNode* parent = nullptr);
	};
}