#include "engine/graphics/Vertex.h"

namespace utad
{
	static const VertexAttribDescription NoneDesc = { -1, GL_FLOAT, 0 };
	static const VertexAttribDescription PositionDesc = { 0, GL_FLOAT, 3 };
	static const VertexAttribDescription NormalDesc = { 1, GL_FLOAT, 3 };
	static const VertexAttribDescription TexCoordsDesc = { 2, GL_FLOAT, 2 };
	static const VertexAttribDescription TangentsDesc = { 3, GL_FLOAT, 3 };
	static const VertexAttribDescription ColorDesc = { 4, GL_FLOAT, 3 };

	static const VertexAttribDescription StandardAttribs[]{
		PositionDesc, NormalDesc, TexCoordsDesc, TangentsDesc, ColorDesc
	};

	static const String StandardAttribsNames[]{
		"Position", "Normal", "TexCoords", "Tangents", "Color"
	};

	const VertexAttribDescription& VertexAttribDescription::of(VertexAttrib attribute)
	{
		const VertexAttribDescription& desc = StandardAttribs[static_cast<int>(attribute) - 1];
		return desc;
	}

	const String& VertexAttribDescription::name(VertexAttrib attribute)
	{
		return StandardAttribsNames[static_cast<int>(attribute) - 1];
	}

	uint VertexAttribDescription::size() const
	{
		int dataSize = 1;

		switch (type)
		{
			case GL_FLOAT:
				dataSize = sizeof(float);
				break;
			case GL_INT:
			case GL_UNSIGNED_INT:
				dataSize = sizeof(int);
				break;
			case GL_BYTE:
			case GL_UNSIGNED_BYTE:
				dataSize = sizeof(byte);
				break;
			case GL_DOUBLE:
				dataSize = sizeof(double);
				break;
		}

		return count * dataSize;
	}

	VertexReader::VertexReader(const void* inputData, size_t size, const ArrayList<VertexAttrib>& attributes, size_t offset)
		: m_Attributes(attributes), m_InputData(static_cast<const byte*>(inputData)), m_DataSize(size), m_Index(offset)
	{
		m_VertexSize = 0;
		for (VertexAttrib attrib : attributes)
			m_VertexSize += VertexAttribDescription::of(attrib).size();
	}

	bool VertexReader::hasNext() const
	{
		return m_Index < m_DataSize;
	}

	Vertex VertexReader::next()
	{
		if (!hasNext()) throw UTAD_EXCEPTION("No more vertices left");

		Vertex vertex = {};

		for (VertexAttrib attrib : m_Attributes)
		{
			setVertexAttribute(vertex, m_InputData + m_Index, attrib);
			m_Index += VertexAttribDescription::of(attrib).size();
		}

		return std::move(vertex);
	}

	void VertexReader::setVertexAttribute(Vertex& vertex, const byte* data, VertexAttrib attribute)
	{
		switch (attribute)
		{
			case VertexAttrib::Position:
				vertex.position = *(const Vector3*)data;
				break;
			case VertexAttrib::Normal:
				vertex.normal = *(const Vector3*)data;
				break;
			case VertexAttrib::TexCoords:
				vertex.texCoords = *(const Vector2*)data;
				break;
			case VertexAttrib::Tangent:
				vertex.tangent = *(const Vector3*)data;
				break;
			case VertexAttrib::Color:
				vertex.color = *(const Color*)data;
				break;
		}
	}

	uint Vertex::stride(const ArrayList<VertexAttrib>& attributes)
	{
		uint stride = 0;
		for (const VertexAttrib& attribute : attributes)
			stride += VertexAttribDescription::of(attribute).size();
		return stride;
	}
}