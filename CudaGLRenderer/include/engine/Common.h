#pragma once

#include <string>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define UTAD_DELETE(ptr) delete ptr; ptr = nullptr
#define UTAD_DELETE_ARRAY(ptr) delete[] ptr; ptr = nullptr

namespace utad 
{
	using String = std::string;
	
	template<typename T>
	using ArrayList = std::vector<T>;

	using uint = unsigned int;

	namespace math
	{
		using namespace glm;
	}

	using Vector2 = glm::vec2;
	using Vector3 = glm::vec3;
	using Vector4 = glm::vec4;
	using Matrix3 = glm::mat3;
	using Matrix4 = glm::mat4;
	using Quaternion = glm::quat;
}