#pragma once

#include "engine/Common.h"

namespace utad
{
	enum class CameraDirection
	{
		Left,
		Right,
		Up,
		Down,
		Forward,
		Backwards
	};

	const float DEFAULT_MAX_FOV = math::radians(90.0f);
	const float DEFAULT_MIN_FOV = math::radians(1.0f);
	const float MIN_PITCH = -89.0f;
	const float MAX_PITCH = 89.0f;
	const float DEFAULT_YAW = -90.0f;
	const float DEFAULT_NEAR_PLANE = 0.1f;
	const float DEFAULT_FAR_PLANE = 1000.0f;
	const float DEFAULT_EXPOSURE = 1.0f;

	class Camera
	{
		friend class Scene;
		friend class RenderInfo;
	private:
		Vector3 m_Position;
		// Axis
		Vector3 m_Forward = {0, 0, -1};
		Vector3 m_Up = {0, 1, 0};
		Vector3 m_Right = {1, 0, 0};
		// Viewport
		Vector4 m_Viewport = {0, 0, 0, 0};
		// Field of view
		Range<float> m_FovRange = {DEFAULT_MIN_FOV, DEFAULT_MAX_FOV};
		float m_Fov = math::clamp(m_FovRange.max / 2.0f, m_FovRange.min, m_FovRange.max);
		// Rotation angles
		float m_Yaw = DEFAULT_YAW;
		float m_Pitch = 0;
		float m_Roll = 0;
		// Sensitivity
		float m_Sensitivity = 0.2f;
		// Planes
		float m_NearPlane = DEFAULT_NEAR_PLANE;
		float m_FarPlane = DEFAULT_FAR_PLANE;
		// HDR
		float m_Exposure = DEFAULT_EXPOSURE;
		// Matrices
		Matrix4 m_ProjectionMatrix;
		Matrix4 m_ViewMatrix;
		Matrix4 m_ProjectionViewMatrix;
		// Movement
		Vector2 m_LastPosition = {0, 0};
		// Rendering
		Vector4 m_ClearColor = {0.15f, 0.15f, 0.15f, 1};
		// Update control
		bool m_Modified = true;
	private:
		Camera();
		~Camera();
	public:
		Camera* lookAt(const Vector2& position);
		Camera* lookAt(float x, float y);
		Camera* move(CameraDirection direction, float amount);
		Camera* zoom(float amount);
		const Matrix4& viewMatrix() const;
		const Matrix4& projectionMatrix() const;
		const Matrix4& projectionViewMatrix() const;
		const Vector4& viewport() const;
		Camera* viewport(const Vector4& viewport);
		const Vector3& position() const;
		Camera* position(const Vector3& position);
		const Vector3& forward() const;
		const Vector3& up() const;
		const Vector3& right() const;
		const Range<float>& fovRange() const;
		Camera* fovRange(const Range<float>& mFovRange);
		float fov() const;
		Camera* fov(float fov);
		float yaw() const;
		Camera* yaw(float mYaw);
		float pitch() const;
		Camera* pitch(float mPitch);
		float roll() const;
		Camera* roll(float mRoll);
		float sensitivity() const;
		Camera* sensitivity(float sensitivity);
		float nearPlane() const;
		Camera* nearPlane(float mNearPlane);
		float farPlane() const;
		Camera* farPlane(float mFarPlane);
		float exposure() const;
		Camera* exposure(float mExposure);
		const Color& clearColor() const;
		Camera* clearColor(const Color& clearColor);
		void updateCameraOrientation();
		void update();
	};

}