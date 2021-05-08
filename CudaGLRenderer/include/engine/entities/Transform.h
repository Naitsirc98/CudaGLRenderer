#pragma once

#include "engine/Common.h"

namespace utad
{
	class Transform
	{
		friend class Entity;
	private:
		Vector3 m_Position;
		Vector3 m_Scale;
		Quaternion m_Rotation;
		Matrix4 m_ModelMatrix;
	private:
		Transform();
	public:
		~Transform();
		Transform(const Transform& other) = delete;
		const Vector3& position() const;
		Vector3& position();
		const Vector3& scale() const;
		Vector3& scale();
		const Quaternion& rotation() const;
		Quaternion& rotation();
		Vector3 eulerAngles() const;
		Transform& eulerAngles(const Vector3& eulerAngles);
		Transform& eulerAngles(Vector3&& eulerAngles);
		Transform& rotate(float radians, const Vector3& axis);
		Transform& rotate(float radians, Vector3&& axis);
		const Matrix4& modelMatrix() const;
		Matrix4& modelMatrix();
	private:
		void computeModelMatrix();
	};

}