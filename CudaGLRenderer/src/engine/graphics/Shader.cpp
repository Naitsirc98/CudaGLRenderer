#include "engine/graphics/Shader.h"
#include "engine/graphics/Texture.h"
#include <iostream>

namespace utad
{
	Shader::Shader(const String& name) : m_Name(name)
	{
		m_Handle = glCreateProgram();
	}

	Shader::~Shader()
	{
		glDeleteProgram(m_Handle);
		m_Handle = NULL;
	}

	Handle Shader::handle() const
	{
		return m_Handle;
	}

	const String& Shader::name() const
	{
		return m_Name;
	}

	Shader& Shader::attach(ShaderStage* stage)
	{
		switch (stage->type)
		{
			case GL_VERTEX_SHADER:
				if (m_VertexStage != nullptr) UTAD_DELETE(m_VertexStage);
				m_VertexStage = stage;
				break;
			case GL_GEOMETRY_SHADER:
				if(m_GeometryStage != nullptr) UTAD_DELETE(m_GeometryStage);
				m_GeometryStage = stage;
				break;
			case GL_FRAGMENT_SHADER:
				if(m_FragmentStage != nullptr) UTAD_DELETE(m_FragmentStage);
				m_FragmentStage = stage;
				break;
		}

		return *this;
	}

	inline GLuint compileShaderStage(ShaderStage* stage)
	{
		if (stage == nullptr) return NULL;

		GLuint shader = glCreateShader(stage->type);

		GLsizei length = stage->sourceCode.size();
		const char* source = stage->sourceCode.c_str();
		glShaderSource(shader, 1, &source, &length);

		glCompileShader(shader);

		GLint success;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			char message[4096];
			GLsizei length;
			glGetShaderInfoLog(shader, sizeof(message), &length, message);
			throw UTAD_EXCEPTION(String("Shader Stage ").append(stage->name).append(" failed to compile: ").append(message).append("\n"));
		}

		return shader;
	}

	inline void attachShaderStage(GLuint program, GLuint shader)
	{
		if (shader != NULL) glAttachShader(program, shader);
	}

	inline void linkProgram(Shader* shader)
	{
		const GLuint program = shader->handle();

		glLinkProgram(program);

		GLint success;
		glGetProgramiv(program, GL_LINK_STATUS, &success);
		if (!success)
		{
			char message[4096];
			GLsizei length;
			glGetProgramInfoLog(program, sizeof(message), &length, message);
			throw UTAD_EXCEPTION(String("Shader Program ").append(shader->name()).append(" failed to compile: ").append(message).append("\n"));
		}
	}

	inline void deleteShaderStage(GLuint program, GLuint stage)
	{
		if (stage == NULL) return;
		glDetachShader(program, stage);
		glDeleteShader(stage);
	}

	void Shader::compile()
	{
		const GLint vertexShader = compileShaderStage(m_VertexStage);
		const GLint geometryShader = compileShaderStage(m_GeometryStage);
		const GLint fragmentShader = compileShaderStage(m_FragmentStage);
		
		attachShaderStage(m_Handle, vertexShader);
		attachShaderStage(m_Handle, geometryShader);
		attachShaderStage(m_Handle, fragmentShader);

		linkProgram(this);

		deleteShaderStage(m_Handle, vertexShader);
		deleteShaderStage(m_Handle, geometryShader);
		deleteShaderStage(m_Handle, fragmentShader);

		UTAD_DELETE(m_VertexStage);
		UTAD_DELETE(m_GeometryStage);
		UTAD_DELETE(m_FragmentStage);
	}

	void Shader::bind()
	{
		glUseProgram(m_Handle);
	}

	void Shader::unbind()
	{
		glUseProgram(0);

		for (size_t unit = 0; unit < m_BoundTextures.size(); ++unit)
		{
			m_BoundTextures[unit]->unbind(unit);
		}

		m_BoundTextures.clear();
		m_TextureUnits.clear();
	}

	void Shader::setTexture(const String& samplerName, Texture2D* texture)
	{
		if (texture == nullptr) return;

		size_t unit;

		if (m_TextureUnits.find(samplerName) != m_TextureUnits.end())
			unit = m_TextureUnits[samplerName];
		else
			unit = m_TextureUnits.size();

		setUniform(samplerName, unit);
		texture->bind(unit);

		m_BoundTextures.push_back(texture);
		m_TextureUnits[samplerName] = unit;
	}

}