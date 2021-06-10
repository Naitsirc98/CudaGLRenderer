#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	enum class VertexAttrib
	{
		None = 0,
		Position,
		Normal,
		TexCoords,
		Tangent,
		Color
	};

	struct VertexAttribDescription
	{
		GLuint location{0};
		GLenum type{0};
		GLsizei count{0};
		uint size() const;

		static const VertexAttribDescription& of(VertexAttrib attribute);
		static const String& name(VertexAttrib attribute);
	};

	struct Vertex
	{
		Vector3 position;
		Vector3 normal;
		Vector2 texCoords;
		Vector3 tangent;
		Color color;

		static uint stride(const ArrayList<VertexAttrib>& attributes);
	};

	class VertexReader
	{
	private:
		ArrayList<VertexAttrib> m_Attributes;
		const byte* m_InputData;
		size_t m_DataSize;
		size_t m_Index{ 0 };
		size_t m_VertexSize;
	public:
		VertexReader(const void* inputData, size_t size, const ArrayList<VertexAttrib>& attributes, size_t offset = 0);
		bool hasNext() const;
		Vertex next();
		static void setVertexAttribute(Vertex& vertex, const byte* data, VertexAttrib attribute);
	};
}