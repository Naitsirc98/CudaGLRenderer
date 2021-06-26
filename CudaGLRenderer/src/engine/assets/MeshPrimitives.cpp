#include "engine/assets/MeshPrimitives.h"
#include "engine/assets/AssetsManager.h"

namespace utad
{

	const GLenum MeshPrimitives::QuadDrawMode = GL_TRIANGLE_STRIP;
	const int MeshPrimitives::QuadVertexCount = 4;
	const GLenum MeshPrimitives::CubeDrawMode = GL_TRIANGLES;
	const int MeshPrimitives::CubeVertexCount = 36;

	const ArrayList<float> MeshPrimitives::s_QuadVertices = {
		-1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f
	};

	const ArrayList<float> MeshPrimitives::s_CubeVertices = {
		// back face
		-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
		 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
		 1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
		 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
		-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
		-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
		// front face
		-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
		 1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
		 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
		 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
		-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
		-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
		// left face
		-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
		-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
		-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
		-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
		-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
		-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
		// right face
		 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
		 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
		 1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
		 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
		 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
		 1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
		// bottom face
		-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
		 1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
		 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
		 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
		-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
		-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
		// top face
		-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
		 1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
		 1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
		 1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
		-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
		-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left  
	};

	Mesh* MeshPrimitives::s_CubeMesh = nullptr;
	Mesh* MeshPrimitives::s_QuadMesh = nullptr;
	Mesh* MeshPrimitives::s_SphereMesh = nullptr;

	Mesh* MeshPrimitives::cube()
	{
		return s_CubeMesh;
	}

	Mesh* MeshPrimitives::quad()
	{
		return s_QuadMesh;
	}

	Mesh* MeshPrimitives::sphere()
	{
		return s_SphereMesh;
	}

	void MeshPrimitives::drawCube(bool bind)
	{
		if(bind) s_CubeMesh->bind();
		glDrawArrays(CubeDrawMode, 0, CubeVertexCount);
		if(bind) s_CubeMesh->unbind();
	}

	void MeshPrimitives::drawQuad(bool bind)
	{
		if(bind) s_QuadMesh->bind();
		glDrawArrays(QuadDrawMode, 0, QuadVertexCount);
		if(bind) s_QuadMesh->unbind();
	}

	void MeshPrimitives::drawSphere(bool bind)
	{
		if(bind) s_SphereMesh->bind();
		glDrawElements(s_SphereMesh->drawMode(), s_SphereMesh->indexCount(),
			s_SphereMesh->indexType(), (void*)s_SphereMesh->indexBufferOffset());
		if(bind) s_SphereMesh->unbind();
	}

	VertexArray* MeshPrimitives::createCubeVAO(ArrayList<Vertex>& vertices, ArrayList<uint>& triangles)
	{
		VertexArray* vao = new VertexArray();
		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = s_CubeVertices.size() * sizeof(float);
		vboAllocInfo.data = (void*)s_CubeVertices.data();
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		ArrayList<VertexAttrib> attributes = { VertexAttrib::Position, VertexAttrib::Normal, VertexAttrib::TexCoords };

		vao->addVertexBuffer(0, vbo, Vertex::stride(attributes));
		vao->setVertexAttributes(0, attributes);
		vao->setDestroyBuffersOnDelete();

		VertexReader reader = VertexReader(s_CubeVertices.data(), s_CubeVertices.size() * sizeof(float), attributes);
		while (reader.hasNext()) vertices.push_back(std::move(reader.next()));

		for (int i = 0; i < vertices.size(); ++i) triangles.push_back(i);

		return vao;
	}

	VertexArray* MeshPrimitives::createQuadVAO(ArrayList<Vertex>& vertices, ArrayList<uint>& triangles)
	{
		VertexArray* vao = new VertexArray();
		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = s_QuadVertices.size() * sizeof(float);
		vboAllocInfo.data = (void*)s_QuadVertices.data();
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		ArrayList<VertexAttrib> attributes = {VertexAttrib::Position, VertexAttrib::Normal, VertexAttrib::TexCoords};

		vao->addVertexBuffer(0, vbo, Vertex::stride(attributes));
		vao->setVertexAttributes(0, attributes);
		vao->setDestroyBuffersOnDelete();

		VertexReader reader(s_QuadVertices.data(), s_QuadVertices.size() * sizeof(float), attributes);
		while (reader.hasNext()) vertices.push_back(std::move(reader.next()));

		for (int i = 0; i < vertices.size(); ++i) triangles.push_back(i);

		return vao;
	}

	VertexArray* MeshPrimitives::createSphereVAO(int xSegments, int ySegments,
		ArrayList<Vertex>& verticesList, ArrayList<uint>& triangles)
	{
		int size = (xSegments + 1) * (ySegments + 1);

		int vertexElementsCount = size * 3 + size * 2 + size * 3;

		float* vertices = new float[vertexElementsCount];
		int index = 0;

		verticesList.reserve(vertexElementsCount / 8);

		for (int y = 0; y <= ySegments; y++)
		{
			for (int x = 0; x <= xSegments; x++)
			{
				float xSeg = (float)x / (float)xSegments;
				float ySeg = (float)y / (float)ySegments;

				float xPos = (float)(std::cos(xSeg * 2 * PI) * std::sin(ySeg * PI));
				float yPos = (float)(std::cos(ySeg * PI));
				float zPos = (float)(std::sin(xSeg * 2 * PI) * std::sin(ySeg * PI));

				// Position
				vertices[index++] = xPos;
				vertices[index++] = yPos;
				vertices[index++] = zPos;
				// Normal
				vertices[index++] = xPos;
				vertices[index++] = yPos;
				vertices[index++] = zPos;
				// UV
				vertices[index++] = xSeg;
				vertices[index++] = ySeg;

				Vertex vertex = {};
				vertex.position = { xPos, yPos, zPos };
				vertex.normal = { xPos, yPos, zPos };
				vertex.texCoords = { xSeg, ySeg };

				verticesList.push_back(std::move(vertex));
			}
		}

		int indicesCount = ySegments * (xSegments + 1) * 2;

		int* indices = new int[indicesCount];
		index = 0;

		for (int y = 0; y < ySegments; y++)
		{
			int index1;
			int index2;
			if (y % 2 == 0)
				for (int x = 0; x <= xSegments; x++)
				{
					index1 = indices[index++] = y * (xSegments + 1) + x;
					index2 = indices[index++] = (y + 1) * (xSegments + 1) + x;
					triangles.push_back(index1);
					triangles.push_back(index2);
				}
			else
				for (int x = xSegments; x >= 0; x--)
				{
					index1 = indices[index++] = (y + 1) * (xSegments + 1) + x;
					index2 = indices[index++] = y * (xSegments + 1) + x;
					triangles.push_back(index1);
					triangles.push_back(index2);
				}
		}

		VertexArray* vao = new VertexArray();
		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = vertexElementsCount * sizeof(float);
		vboAllocInfo.data = vertices;
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		vao->addVertexBuffer(0, vbo, 8 * sizeof(float));
		vao->setVertexAttributes(0, { VertexAttrib::Position, VertexAttrib::Normal, VertexAttrib::TexCoords });

		IndexBuffer* ibo = new IndexBuffer();

		BufferAllocInfo iboAllocInfo = {};
		iboAllocInfo.size = indicesCount * sizeof(int);
		iboAllocInfo.data = indices;
		iboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		ibo->allocate(std::move(iboAllocInfo));

		vao->setIndexBuffer(ibo);

		vao->setDestroyBuffersOnDelete();

		delete[] vertices;
		delete[] indices;

		return vao;
	}

	Mesh* MeshPrimitives::createCubeMesh()
	{
		ArrayList<Vertex> vertices;
		ArrayList<uint> triangles;
		VertexArray* vao = createCubeVAO(vertices, triangles);

		Mesh* mesh = new Mesh(vao);
		mesh->m_DrawMode = CubeDrawMode;
		mesh->m_Vertices = std::move(vertices);
		mesh->m_Indices = std::move(triangles);

		return mesh;
	}

	Mesh* MeshPrimitives::createQuadMesh()
	{
		ArrayList<Vertex> vertices;
		ArrayList<uint> triangles;
		VertexArray* vao = createQuadVAO(vertices, triangles);

		Mesh* mesh = new Mesh(vao);
		mesh->m_DrawMode = QuadDrawMode;
		mesh->m_Vertices = std::move(vertices);
		mesh->m_Indices = std::move(triangles);

		return mesh;
	}

	Mesh* MeshPrimitives::createSphereMesh(int xSegments, int ySegments)
	{
		ArrayList<Vertex> vertices;
		ArrayList<uint> triangles;
		VertexArray* vao = createSphereVAO(xSegments, ySegments, vertices, triangles);

		Mesh* mesh = new Mesh(vao);
		mesh->m_IndexBufferOffset = 0;
		mesh->m_IndexType = GL_UNSIGNED_INT;
		mesh->m_IndexCount = vao->indexBuffer()->size() / sizeof(unsigned int);
		mesh->m_DrawMode = GL_TRIANGLE_STRIP;
		mesh->m_Vertices = std::move(vertices);
		mesh->m_Indices = std::move(triangles);

		return mesh;
	}

	void MeshPrimitives::initMeshes()
	{
		s_CubeMesh = createCubeMesh();
		s_QuadMesh = createQuadMesh();
		s_SphereMesh = createSphereMesh(128, 128);
	}

	void MeshPrimitives::destroyMeshes()
	{
		UTAD_DELETE(s_CubeMesh);
		UTAD_DELETE(s_QuadMesh);
		UTAD_DELETE(s_SphereMesh);
	}
}