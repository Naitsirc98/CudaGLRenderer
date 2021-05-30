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
	};
}