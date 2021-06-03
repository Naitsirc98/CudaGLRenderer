#include "engine/scene/Camera.h"
#include "engine/graphics/Window.h"
#include "engine/events/Input.h"

namespace utad
{
	Camera::Camera()
	{
		int width = Window::get().width();
		int height = Window::get().height();
		m_Viewport.z = static_cast<float>(width);
		m_Viewport.w = static_cast<float>(height);
		m_LastPosition = Input::getMousePosition();
		m_ProjectionMatrix = Matrix4(1.0f);
		m_ViewMatrix = Matrix4(1.0f);
	}

	Camera::~Camera()
	{
	}

	Camera* Camera::lookAt(const Vector2& position)
	{
		if (position == m_LastPosition) return this;

		float xOffset = position.x - m_LastPosition.x;
		float yOffset = m_LastPosition.y - position.y;

		m_LastPosition = position;

		float fov = math::max(this->fov(), 1.0f);

		xOffset *= m_Sensitivity / (m_FovRange.max / fov);
		yOffset *= m_Sensitivity / (m_FovRange.max / fov);

		m_Yaw += xOffset;
		m_Pitch = math::clamp(yOffset + m_Pitch, MIN_PITCH, MAX_PITCH);

		return this;
	}

	Camera* Camera::lookAt(float x, float y)
	{
		return lookAt({ x, y });
	}

	Camera* Camera::move(CameraDirection direction, float amount)
	{
		switch (direction)
		{
		case CameraDirection::Left:
			m_Position += (-m_Right * amount);
			break;
		case CameraDirection::Right:
			m_Position += (m_Right * amount);
			break;
		case CameraDirection::Up:
			m_Position += (m_Up * amount);
			break;
		case CameraDirection::Down:
			m_Position += (-m_Up * amount);
			break;
		case CameraDirection::Forward:
			m_Position += (m_Forward * amount);
			break;
		case CameraDirection::Backwards:
			m_Position += (-m_Forward * amount);
			break;
		}

		return this;
	}

	Camera* Camera::zoom(float amount)
	{
		fov(fov() - math::radians(amount));
		return this;
	}

	const Matrix4& Camera::viewMatrix() const
	{
		return m_ViewMatrix;
	}

	const Matrix4& Camera::projectionMatrix() const
	{
		return m_ProjectionMatrix;
	}

	const Matrix4& Camera::projectionViewMatrix() const
	{
		return m_ProjectionViewMatrix;
	}

	const Vector4& Camera::viewport() const
	{
		return m_Viewport;
	}

	Camera* Camera::viewport(const Vector4& viewport)
	{
		m_Viewport = viewport;
		return this;
	}

	const Vector3& Camera::position() const
	{
		return m_Position;
	}

	Camera* Camera::position(const Vector3& position)
	{
		m_Position = position;
		return this;
	}

	const Vector3& Camera::forward() const
	{
		return m_Forward;
	}

	const Vector3& Camera::up() const
	{
		return m_Up;
	}

	const Vector3& Camera::right() const
	{
		return m_Right;
	}

	const Range<float>& Camera::fovRange() const
	{
		return m_FovRange;
	}

	Camera* Camera::fovRange(const Range<float>& mFov)
	{
		m_FovRange = mFov;
		return this;
	}

	float Camera::fov() const
	{
		return m_Fov;
	}

	Camera* Camera::fov(float fov)
	{
		m_Fov = math::clamp(fov, m_FovRange.min, m_FovRange.max);
		return this;
	}

	float Camera::yaw() const
	{
		return m_Yaw;
	}

	Camera* Camera::yaw(float mYaw)
	{
		m_Yaw = mYaw;
		return this;
	}

	float Camera::pitch() const
	{
		return m_Pitch;
	}

	Camera* Camera::pitch(float mPitch)
	{
		m_Pitch = math::clamp(mPitch, MIN_PITCH, MAX_PITCH);
		return this;
	}

	float Camera::roll() const
	{
		return m_Roll;
	}

	Camera* Camera::roll(float mRoll)
	{
		m_Roll = mRoll;
		return this;
	}

	float Camera::sensitivity() const
	{
		return m_Sensitivity;
	}

	Camera* Camera::sensitivity(float sensitivity)
	{
		m_Sensitivity = sensitivity;
		return this;
	}

	float Camera::nearPlane() const
	{
		return m_NearPlane;
	}

	Camera* Camera::nearPlane(float mNearPlane)
	{
		m_NearPlane = mNearPlane;
		return this;
	}

	float Camera::farPlane() const
	{
		return m_FarPlane;
	}

	Camera* Camera::farPlane(float mFarPlane)
	{
		m_FarPlane = mFarPlane;
		return this;
	}

	float Camera::exposure() const
	{
		return m_Exposure;
	}

	Camera* Camera::exposure(float mExposure)
	{
		m_Exposure = mExposure;
		return this;
	}

	const Color& Camera::clearColor() const
	{
		return m_ClearColor;
	}

	Camera* Camera::clearColor(const Color& clearColor)
	{
		m_ClearColor = clearColor;
		return this;
	}

	void Camera::updateCameraOrientation()
	{
		const float yaw = math::radians(m_Yaw);
		const float pitch = math::radians(m_Pitch);
		m_Forward.x = cos(yaw) * cos(pitch);
		m_Forward.y = sin(pitch);
		m_Forward.z = sin(yaw) * cos(pitch);
		m_Forward = math::normalize(m_Forward);
		m_Right = math::normalize(cross(m_Forward, { 0, 1, 0 }));
		m_Up = math::normalize(cross(m_Right, m_Forward));
	}

	static Matrix4 getViewMatrix(const Vector3& pos, const Vector3& fwd, const Vector3& up)
	{
		return math::lookAt(pos, pos + fwd, up);
	}

	static Matrix4 getProjectionMatrix(Camera* camera)
	{
		const float aspect = camera->viewport().w == 0 ? 0 : camera->viewport().z / camera->viewport().w;
		return math::perspective(camera->fov(), aspect, camera->nearPlane(), camera->farPlane());
	}

	void Camera::update()
	{
		updateCameraOrientation();
		m_ViewMatrix = getViewMatrix(position(), forward(), up());
		m_ProjectionMatrix = getProjectionMatrix(this);
		m_ProjectionViewMatrix = m_ProjectionMatrix * m_ViewMatrix;
		m_Modified = false;
	}
}