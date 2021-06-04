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
		mutable Map<String, int> m_UniformLocations;
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
			glUniform1i(location(name), value);
		}

		template<>
		void setUniform<float>(const String& name, const float& value)
		{
			glUniform1f(location(name), value);
		}

		template<>
		void setUniform<int>(const String& name, const int& value)
		{
			glUniform1i(location(name), value);
		}

		template<>
		void setUniform<Vector3>(const String& name, const Vector3& value)
		{
			glUniform3fv(location(name), 1, math::value_ptr(value));
		}

		template<>
		void setUniform<Vector4>(const String& name, const Vector4& value)
		{
			glUniform4fv(location(name), 1, math::value_ptr(value));
		}

		template<>
		void setUniform<Matrix4>(const String& name, const Matrix4& value)
		{
			glUniformMatrix4fv(location(name), 1, false, math::value_ptr(value));
		}

	private:
		inline int location(const String& name) const
		{
			if (m_UniformLocations.find(name) != m_UniformLocations.end()) return m_UniformLocations[name];
			const int location = glGetUniformLocation(m_Handle, name.c_str());
			if (location < 0)
			{
				std::cout << "[WARNING][Shader " << m_Name << "]: uniform "
					<< name << " returned invalid location: " << location << std::endl;
			}
			m_UniformLocations[name] = location;
			return location;
		}
	};
}