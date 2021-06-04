#pragma once

#include "engine/graphics/Graphics.h"

namespace utad
{
	class Primitives
	{
	public:
		static const GLenum quadDrawMode;
		static const int quadVertexCount;
		static const ArrayList<float> quadVertices;

		static const GLenum cubeDrawMode;
		static const int cubeVertexCount;
		static const ArrayList<float> cubeVertices;

		static VertexArray* createCubeVAO();

		static VertexArray* createQuadVAO();
	};

}