#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#define UTAD_DELETE(ptr) delete ptr; ptr = nullptr
#define UTAD_DELETE_ARRAY(ptr) delete[] ptr; ptr = nullptr


namespace utad 
{
	using String = std::string;
	
	template<typename T>
	using Vector = std::vector<T>;

	using uint = unsigned int;

	using Vector2 = glm::vec2;
	using Vector3 = glm::vec3;
	using Vector4 = glm::vec4;
	using Matrix3 = glm::mat3;
	using Matrix4 = glm::mat4;
}