#pragma once

#include "engine/graphics/Graphics.h"
#include "Mesh.h"

namespace utad
{
	class MeshPrimitives
	{
		friend class AssetsManager;

	public:
		static const GLenum QuadDrawMode;
		static const int QuadVertexCount;
		static const GLenum CubeDrawMode;
		static const int CubeVertexCount;

	private:
		static const ArrayList<float> s_QuadVertices;
		static const ArrayList<float> s_CubeVertices;

		static Mesh* s_CubeMesh;
		static Mesh* s_QuadMesh;
		static Mesh* s_SphereMesh;

	public:
		static Mesh* cube();
		static Mesh* quad();
		static Mesh* sphere();

		static void drawCube(bool bind = true);
		static void drawQuad(bool bind = true);
		static void drawSphere(bool bind = true);
	
	private:
		static VertexArray* createCubeVAO(ArrayList<Vertex>& vertices, ArrayList<uint>& triangles);
		static VertexArray* createQuadVAO(ArrayList<Vertex>& vertices, ArrayList<uint>& triangles);
		static VertexArray* createSphereVAO(int xSegments, int ySegments, ArrayList<Vertex>& vertices, ArrayList<uint>& triangles);

		static Mesh* createCubeMesh();
		static Mesh* createQuadMesh();
		static Mesh* createSphereMesh(int xSegments, int ySegments);

		static void initMeshes();
		static void destroyMeshes();
	};

}