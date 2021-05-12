#include "engine/Time.h"
#include <GLFW/glfw3.h>

namespace utad
{
	float Time::s_DeltaTime = 0.0f;
	unsigned int Time::s_Frame = 0;


	float Time::time() noexcept
	{
		return static_cast<float>(glfwGetTime());
	}

	float Time::deltaTime() noexcept
	{
		return s_DeltaTime;
	}

	unsigned int Time::frame() noexcept
	{
		return s_Frame;
	}

}