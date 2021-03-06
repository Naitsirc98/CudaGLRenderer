#include "engine/entities/Transform.h"
#include "engine/entities/Entity.h"
#include <glm/gtx/matrix_decompose.hpp>

namespace utad
{

	Transform::Transform()
		: m_Position({ 0, 0, 0 }), m_Scale({1, 1, 1}), m_Rotation({0, 0, 0}), m_ModelMatrix(Matrix4(1.0f))
	{
	}

	Transform::~Transform()
	{
		m_Entity = nullptr;
	}

	Entity& Transform::entity() const
	{
		return *m_Entity;
	}

	const Vector3& Transform::position() const
	{
		return m_Position;
	}

	Vector3& Transform::position()
	{
		return m_Position;
	}

	const Vector3& Transform::scale() const
	{
		return m_Scale;
	}

	Vector3& Transform::scale()
	{
		return m_Scale;
	}

	const Quaternion& Transform::rotation() const
	{
		return m_Rotation;
	}

	Quaternion& Transform::rotation()
	{
		return m_Rotation;
	}

	Vector3 Transform::eulerAngles() const
	{
		return math::eulerAngles(m_Rotation);
	}

	Transform& Transform::eulerAngles(const Vector3& eulerAngles)
	{
		m_Rotation = Quaternion(eulerAngles);
		return *this;
	}

	Transform& Transform::eulerAngles(Vector3&& eulerAngles)
	{
		m_Rotation = Quaternion(std::move(eulerAngles));
		return *this;
	}

	Transform& Transform::rotate(float radians, const Vector3& axis)
	{
		m_Rotation = math::angleAxis(radians, normalize(axis));
		return *this;
	}

	Transform& Transform::rotate(float radians, Vector3&& axis)
	{
		m_Rotation = math::angleAxis(radians, normalize(axis));
		return *this;
	}

	bool Transform::ignoreParent() const
	{
		return m_IgnoreParent;
	}

	Transform& Transform::ignoreParent(bool ignore)
	{
		m_IgnoreParent = ignore;
		return *this;
	}

	const Matrix4& Transform::modelMatrix() const
	{
		return m_ModelMatrix;
	}

	Matrix4& Transform::modelMatrix()
	{
		return m_ModelMatrix;
	}

	Transform& Transform::modelMatrix(const Matrix4& modelMatrix)
	{
		m_ModelMatrix = modelMatrix;
		Vector3 skew;
		Vector4 perspective;
		math::decompose(modelMatrix, m_Scale, m_Rotation, m_Position, skew, perspective);
		return *this;
	}

	void Transform::computeModelMatrix()
	{
		const Matrix4 translate = math::translate(m_Position);
		const Matrix4 scale = math::scale(m_Scale);
		const Matrix4 rotate = math::toMat4(m_Rotation);

		m_ModelMatrix = translate * scale * rotate;

		if (!m_IgnoreParent && m_Entity->hasParent())
		{
			m_ModelMatrix = m_Entity->parent()->transform().modelMatrix();
		}
	}
}