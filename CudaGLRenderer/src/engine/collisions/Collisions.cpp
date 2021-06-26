#include "engine/collisions/Collisions.h"

namespace utad
{
	bool Triangle::testCollision(const Ray& ray, Material* material, Collision& collision) const
	{
		static const float kEpsilon = 1e-8;

		// compute plane's normal
		Vector3 v0 = this->v0->position;
		Vector3 v1 = this->v1->position;
		Vector3 v2 = this->v2->position;
		
		Vector3 v0v1 = v1 - v0;
		Vector3 v0v2 = v2 - v0;
		Vector3 N = math::cross(v0v1, v0v2); // N
		Vector3 pvec = math::cross(ray.direction, v0v2);
		float det = math::dot(v0v1, pvec);

		// ray and triangle are parallel if det is close to 0
		if (fabs(det) < kEpsilon) return false;

		float invDet = 1 / det;

		Vector3 tvec = ray.origin - v0;
		float u = math::dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) return false;

		Vector3 qvec = math::cross(tvec, v0v1);
		float v = math::dot(ray.direction, qvec) * invDet;
		if (v < 0 || u + v > 1) return false;

		float t = math::dot(v0v2, qvec) * invDet;

		Vector3 P = ray.origin + t * ray.direction;

		collision.vertex.color = this->v0->color;
		collision.vertex.normal = (1.0f - u - v) * this->v0->normal + u * this->v1->normal + v * this->v2->normal;
		collision.vertex.position = math::vec4(P, 1.0f);
		collision.vertex.texCoords = (1.0f - u - v) * this->v0->texCoords + u * this->v1->texCoords + v * this->v2->texCoords;
		collision.distance = math::length(P - ray.origin);
		collision.material = material;

		return true; // this ray hits the triangle
	}

	AABB::AABB()
	{
		UTAD_DELETE(m_TransformedVertices);
	}

	AABB::AABB(Mesh* mesh)
	{
		m_IsRoot = true;
		m_Vertices = const_cast<ArrayList<Vertex>*>(&mesh->vertices());
		m_TransformedVertices = new ArrayList<Vertex>();

		m_VertexIds = const_cast<ArrayList<uint>*>(&mesh->indices());

		for (const Vertex& vertex : mesh->vertices())
			m_TransformedVertices->push_back(vertex);


		m_Triangles.reserve(mesh->indices().size() / 3);
		for (uint i = 0;i < mesh->indices().size();i += 3)
		{
			Triangle triangle;
			triangle.v0 = &(*m_Vertices)[mesh->indices()[i]];
			triangle.v1 = &(*m_Vertices)[mesh->indices()[i + 1]];
			triangle.v2 = &(*m_Vertices)[mesh->indices()[i + 2]];

			m_Triangles.push_back(std::move(triangle));
		}

		m_TransformedTriangles.reserve(m_Triangles.size());
		for (uint i = 0; i < mesh->indices().size(); i += 3)
		{
			Triangle triangle;
			triangle.v0 = &(*m_TransformedVertices)[mesh->indices()[i]];
			triangle.v1 = &(*m_TransformedVertices)[mesh->indices()[i + 1]];
			triangle.v2 = &(*m_TransformedVertices)[mesh->indices()[i + 2]];
		
			m_TransformedTriangles.push_back(std::move(triangle));
		}

		resize();
		subdivide();
	}

	AABB::~AABB()
	{
	}

	void AABB::update(const Matrix4& transformation)
	{
		for (AABB* child : m_Children) delete child;
		m_Children.clear();

		const Matrix4 normalMatrix = math::inverse(math::transpose(transformation));

		for (uint index : *m_VertexIds)
		{
			Vertex& vertex = (*m_TransformedVertices)[index];
			vertex.position = Vector3(transformation * Vector4((*m_Vertices)[index].position, 1.0f));
			vertex.normal = normalMatrix * Vector4((*m_Vertices)[index].normal, 1.0);
		}

		resize();
		subdivide();
	}

	void AABB::subdivide()
	{
		if (m_TransformedTriangles.empty()) return;

		AABB* subAABB[8]{nullptr};
		updateOctree(subAABB);

		for (const Triangle& triangle : m_TransformedTriangles)
		{
			Vector3 v0 = triangle.v0->position;
			Vector3 v1 = triangle.v1->position;
			Vector3 v2 = triangle.v2->position;

			Vector3 p;
			p.x = (v0.x + v1.x + v2.x) / 3.0;
			p.y = (v0.y + v1.y + v2.y) / 3.0;
			p.z = (v0.z + v1.z + v2.z) / 3.0;

			bool shouldExit = false;
			int bb = 0;

			while (!shouldExit && (bb < 8))
			{
				if ((subAABB[bb]) && subAABB[bb]->testPoint(p))
				{
					subAABB[bb]->m_TransformedTriangles.push_back(triangle);
					shouldExit = true;
				}
				++bb;
			}

			//if (!shouldExit) std::cout << "ERROR AABB 113\n";
		}

		for (int i = 0; i < 8; ++i)
		{
			if (subAABB[i] && (subAABB[i]->m_TransformedTriangles.size() > 0) && (subAABB[i]->m_TransformedTriangles.size() != m_TransformedTriangles.size()))
				m_Children.push_back(subAABB[i]);
			else
				delete subAABB[i];
		}

		for (AABB* child : m_Children)
		{
			child->resize();
			child->subdivide();
		}
	}

	void AABB::resize()
	{
		m_Min = Vector3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX);
		m_Max = Vector3(FLOAT_MIN, FLOAT_MIN, FLOAT_MIN);

		for (const Triangle& triangle : m_TransformedTriangles)
		{
			for (int vt = 0; vt < 3; vt++)
			{
				const Vector3& v = triangle.vertices[vt]->position;

				if (v.x < m_Min.x)	m_Min.x = v.x;
				if (v.y < m_Min.y)	m_Min.y = v.y;
				if (v.z < m_Min.z)	m_Min.z = v.z;
				if (v.x > m_Max.x)	m_Max.x = v.x;
				if (v.y > m_Max.y)	m_Max.y = v.y;
				if (v.z > m_Max.z)	m_Max.z = v.z;
			}
		}
	}

	void AABB::getCollisions(const Ray& ray, SortedMap<float, Collision>& collisions)
	{

		if (!testAABB(ray)) return;

		if (!m_Children.empty())
		{
			for (AABB* child : m_Children)
				child->getCollisions(ray, collisions);
		}
		else
		{
			Collision collision;
			for (const Triangle& triangle : m_TransformedTriangles)
			{
				if (triangle.testCollision(ray, m_Material, collision))
					collisions[collision.distance] = collision;
			}
		}
	}

	bool AABB::testAABB(const Ray& r)
	{
		float tmin = (m_Min.x - r.origin.x) / r.direction.x;
		float tmax = (m_Max.x - r.origin.x) / r.direction.x;

		if (tmin > tmax) std::swap(tmin, tmax);

		float tymin = (m_Min.y - r.origin.y) / r.direction.y;
		float tymax = (m_Max.y - r.origin.y) / r.direction.y;

		if (tymin > tymax) std::swap(tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (m_Min.z - r.origin.z) / r.direction.z;
		float tzmax = (m_Max.z - r.origin.z) / r.direction.z;

		if (tzmin > tzmax) std::swap(tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;

		if (tzmax < tmax)
			tmax = tzmax;

		return true;
	}

	bool AABB::testPoint(const Vector3& p)
	{
		return (p.x >= m_Min[0] && p.x <= m_Max[0]) &&
			   (p.y >= m_Min[1] && p.y <= m_Max[1]) &&
			   (p.z >= m_Min[2] && p.z <= m_Max[2]);
	}

	void AABB::updateOctree(AABB* subAABB[8])
	{
		Vector3 step = Vector3((m_Max.x - m_Min.x) / 2.0f, (m_Max.y - m_Min.y) / 2.0f, (m_Max.z - m_Min.z) / 2.0f);

		glm::vec3 margin = step / 2.0f;
		if (step.x == 0) step.x = 0.0001f;
		if (step.y == 0) step.y = 0.0001f;
		if (step.z == 0) step.z = 0.0001f;
		for (int i = 0; i < 8; i++)
		{
			subAABB[i] = new AABB();
			subAABB[i]->m_Material = m_Material;
		}
		//dlb
		subAABB[0]->m_Min.x = m_Min.x;
		subAABB[0]->m_Min.y = m_Min.y;
		subAABB[0]->m_Min.z = m_Min.z;
		subAABB[0]->m_Max.x = m_Min.x + step.x;
		subAABB[0]->m_Max.y = m_Min.y + step.y;
		subAABB[0]->m_Max.z = m_Min.z + step.z;
		//drb
		subAABB[1]->m_Min.x = m_Min.x + step.x;
		subAABB[1]->m_Min.y = m_Min.y;
		subAABB[1]->m_Min.z = m_Min.z;
		subAABB[1]->m_Max.x = m_Max.x;
		subAABB[1]->m_Max.y = m_Min.y + step.y;
		subAABB[1]->m_Max.z = m_Min.z + step.z;
		//ulb
		subAABB[2]->m_Min.x = m_Min.x;
		subAABB[2]->m_Min.y = m_Min.y + step.y;
		subAABB[2]->m_Min.z = m_Min.z;
		subAABB[2]->m_Max.x = m_Min.x + step.x;
		subAABB[2]->m_Max.y = m_Max.y;
		subAABB[2]->m_Max.z = m_Min.z + step.z;
		//urb
		subAABB[3]->m_Min.x = m_Min.x + step.x;
		subAABB[3]->m_Min.y = m_Min.y + step.y;
		subAABB[3]->m_Min.z = m_Min.z;
		subAABB[3]->m_Max.x = m_Max.x;
		subAABB[3]->m_Max.y = m_Max.y;
		subAABB[3]->m_Max.z = m_Min.z + step.z;
		//dlf
		subAABB[4]->m_Min.x = m_Min.x;
		subAABB[4]->m_Min.y = m_Min.y;
		subAABB[4]->m_Min.z = m_Min.z + step.z;
		subAABB[4]->m_Max.x = m_Min.x + step.x;
		subAABB[4]->m_Max.y = m_Min.y + step.y;
		subAABB[4]->m_Max.z = m_Max.z;
		//drf
		subAABB[5]->m_Min.x = m_Min.x + step.x;
		subAABB[5]->m_Min.y = m_Min.y;
		subAABB[5]->m_Min.z = m_Min.z + step.z;
		subAABB[5]->m_Max.x = m_Max.x;
		subAABB[5]->m_Max.y = m_Min.y + step.y;
		subAABB[5]->m_Max.z = m_Max.z;
		//ulf
		subAABB[6]->m_Min.x = m_Min.x;
		subAABB[6]->m_Min.y = m_Min.y + step.y;
		subAABB[6]->m_Min.z = m_Min.z + step.z;
		subAABB[6]->m_Max.x = m_Min.x + step.x;
		subAABB[6]->m_Max.y = m_Max.y;
		subAABB[6]->m_Max.z = m_Max.z;
		//urf
		subAABB[7]->m_Min.x = m_Min.x + step.x;
		subAABB[7]->m_Min.y = m_Min.y + step.y;
		subAABB[7]->m_Min.z = m_Min.z + step.z;
		subAABB[7]->m_Max.x = m_Max.x;
		subAABB[7]->m_Max.y = m_Max.y;
		subAABB[7]->m_Max.z = m_Max.z;
	}
}