#include "engine/graphics/Shader.h"
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
			char message[512];
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
			char message[512];
			GLsizei length;
			glGetProgramInfoLog(program, sizeof(message), &length, message);
			throw UTAD_EXCEPTION(String("Shader Program ").append(shader->name()).append(" failed to compile: ").append(message).append("\n"));
		}
	}

	inline void deleteShaderStage(GLuint program, GLuint stage)
	{
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

		m_VertexStage = m_GeometryStage = m_FragmentStage = nullptr;
	}

}