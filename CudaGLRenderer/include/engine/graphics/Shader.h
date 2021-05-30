#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	struct ShaderStage
	{
		GLenum type;
		String name;
		String sourceCode;
	};

	class Shader
	{
	private:
		Handle m_Handle;
		String m_Name;
		ShaderStage* m_VertexStage{nullptr};
		ShaderStage* m_GeometryStage{nullptr};
		ShaderStage* m_FragmentStage{nullptr};
	public:
		Shader(const String& name);
		~Shader();
		Handle handle() const;
		const String& name() const;
		Shader& attach(ShaderStage* stage);
		void compile();
		void bind();
		void unbind();

		template<typename T>
		void setUniform(const String& name, const T& value) {}

		template<>
		void setUniform<float>(const String& name, const float& value)
		{
			glUniform1f(glGetUniformLocation(m_Handle, name.c_str()), value);
		}

		template<>
		void setUniform<int>(const String& name, const int& value)
		{
			glUniform1i(glGetUniformLocation(m_Handle, name.c_str()), value);
		}

		template<>
		void setUniform<Vector3>(const String& name, const Vector3& value)
		{
			glUniform3f(glGetUniformLocation(m_Handle, name.c_str()), value.x, value.y, value.z);
		}

		template<>
		void setUniform<Vector4>(const String& name, const Vector4& value)
		{
			glUniform4f(glGetUniformLocation(m_Handle, name.c_str()), value.x, value.y, value.z, value.w);
		}

		template<>
		void setUniform<Matrix4>(const String& name, const Matrix4& value)
		{
			glUniformMatrix4fv(glGetUniformLocation(m_Handle, name.c_str()), 1, false, math::value_ptr(value));
		}
	};
}