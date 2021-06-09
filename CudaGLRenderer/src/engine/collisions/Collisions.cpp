#include "engine/collisions/Collisions.h"

namespace utad
{
	bool Triangle::testCollision(const Ray& ray, Collision& collision)
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

		return true; // this ray hits the triangle
	}

	AABB::AABB()
	{
	}

	AABB::AABB(Mesh* mesh)
	{
	}

	AABB::~AABB()
	{
	}

	void AABB::update(const Matrix4& transformation)
	{
	}

	void AABB::subdivide()
	{
	}

	void AABB::resize()
	{
	}

	void AABB::getCollisions(const Ray& ray, SortedMap<float, Collision>& collisions)
	{
	}

	bool AABB::testAABB(const Ray& ray)
	{
		return false;
	}

	bool AABB::testPoint(const Vector3& point)
	{
		return false;
	}

}