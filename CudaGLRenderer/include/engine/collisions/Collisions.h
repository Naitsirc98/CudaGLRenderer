#pragma once

#include "engine/Common.h"
#include "engine/assets/Mesh.h"
#include "engine/assets/Material.h"

#define FLOAT_MIN -std::numeric_limits<float>::max()
#define FLOAT_MAX  std::numeric_limits<float>::max()

namespace utad
{
	struct Collision
	{
		Vertex vertex;
		float distance;
		Material* material;
	};

	struct Ray
	{
		Vector3 origin;
		Vector3 direction;
		float distance;
		bool inside;
	};

	struct Triangle
	{
		union
		{
			struct { Vertex* v0, *v1, *v2; };
			Vertex* vertices[3];
		};

		bool testCollision(const Ray& ray, Material* material, Collision& collision) const;
	};

	class AABB
	{
	private:
		Vector3 m_Min{FLOAT_MAX};
		Vector3 m_Max{FLOAT_MIN};
		ArrayList<AABB*> m_Children;
		ArrayList<Vertex>* m_Vertices{nullptr};
		ArrayList<Triangle> m_Triangles;
		ArrayList<uint>* m_VertexIds{nullptr};
		ArrayList<Vertex>* m_TransformedVertices{nullptr};
		ArrayList<Triangle> m_TransformedTriangles;
		Material* m_Material;
		bool m_IsRoot{false};
	public:
		AABB();
		AABB(Mesh* mesh);
		~AABB();
		void update(const Matrix4& transformation);
		void subdivide();
		void resize();
		void getCollisions(const Ray& ray, SortedMap<float, Collision>& collisions);
		bool testAABB(const Ray& ray);
		bool testPoint(const Vector3& point);
	private:
		void updateOctree(AABB* subAABB[8]);
	};
}