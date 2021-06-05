#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <deque>
#include <unordered_set>
#include <functional>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define UTAD_DELETE(ptr) delete ptr; ptr = nullptr
#define UTAD_DELETE_ARRAY(ptr) delete[] ptr; ptr = nullptr

#define UTAD_EXCEPTION(msg) std::runtime_error(String((msg)).append(" at\n\t").append(__FILE__).append("(").append(std::to_string(__LINE__)).append(")"))

namespace utad 
{
	using String = std::string;
	
	template<typename T>
	using ArrayList = std::vector<T>;

	template<typename T, size_t Size>
	using Array = std::array<T, Size>;

	template<typename K, typename V>
	using Map = std::unordered_map<K, V>;

	template<typename K, typename V>
	using SortedMap = std::map<K, V>;

	template<typename T>
	using Set = std::unordered_set<T>;

	template<typename T>
	using Stack = std::deque<T>;

	template<typename T>
	using Queue = std::deque<T>;

	using byte = char;
	using uint = unsigned int;

	template<typename R, typename ...Args>
	using Function = std::function<R(Args...)>;

	using KeyModifiersBitMask = int;

	namespace math
	{
		using namespace glm;
	}

	using Vector2i = glm::ivec2;
	using Vector2 = glm::vec2;
	using Vector3 = glm::vec3;
	using Vector4 = glm::vec4;
	using Matrix3 = glm::mat3;
	using Matrix4 = glm::mat4;
	using Quaternion = glm::quat;

	using Color = Vector4;

	namespace colors
	{
		const Color WHITE = {1, 1, 1, 1};
		const Color BLACK = {0, 0, 0, 1};
		const Color RED = {1, 0, 0, 1};
		const Color GREEN = {0, 1, 0, 1};
		const Color BLUE = {0, 0, 1, 1};
	}

	template<typename T>
	struct Range
	{
		T min;
		T max;
	};
}