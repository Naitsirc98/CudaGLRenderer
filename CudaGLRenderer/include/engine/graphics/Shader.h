#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	class Texture;

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
		Map<String, size_t> m_TextureUnits;
		ArrayList<Texture*> m_BoundTextures;
	public:
		Shader(const String& name);
		~Shader();
		Handle handle() const;
		const String& name() const;
		Shader& attach(ShaderStage* stage);
		void compile();
		void bind();
		void unbind();
		void setTexture(const String& samplerName, Texture* texture);

		template<typename T>
		void setUniform(const String& name, const T& value) {}

		template<>
		void setUniform<bool>(const String& name, const bool& value)
		{
			glUniform1i(glGetUniformLocation(m_Handle, name.c_str()), value);
		}

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
			glUniform3fv(glGetUniformLocation(m_Handle, name.c_str()), 1, math::value_ptr(value));
		}

		template<>
		void setUniform<Vector4>(const String& name, const Vector4& value)
		{
			glUniform4fv(glGetUniformLocation(m_Handle, name.c_str()), 1, math::value_ptr(value));
		}

		template<>
		void setUniform<Matrix4>(const String& name, const Matrix4& value)
		{
			glUniformMatrix4fv(glGetUniformLocation(m_Handle, name.c_str()), 1, false, math::value_ptr(value));
		}
	};
}