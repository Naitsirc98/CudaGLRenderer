#pragma once

#include "engine/graphics/Graphics.h"
#include "Mesh.h"

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

		static VertexArray* createSphereVAO(int xSegments, int ySegments);

		static Mesh* createSphereMesh(int xSegments, int ySegments);
	};

}