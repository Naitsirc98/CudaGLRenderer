#include "engine/assets/Primitives.h"
#include "engine/assets/AssetsManager.h"

namespace utad
{
	const GLenum Primitives::quadDrawMode = GL_TRIANGLE_STRIP;
	const int Primitives::quadVertexCount = 4;
	const ArrayList<float> Primitives::quadVertices = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f
	};

	const GLenum Primitives::cubeDrawMode = GL_TRIANGLES;
	const int Primitives::cubeVertexCount = 36;
	const ArrayList<float> Primitives::cubeVertices = {
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


#define PI 3.141592653589f

	VertexArray* Primitives::createSphereVAO(int xSegments, int ySegments)
	{
		int size = (xSegments + 1) * (ySegments + 1);

		int vertexElementsCount = size * 3 + size * 2 + size * 3;

		float* vertices = new float[vertexElementsCount];
		int index = 0;

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
			}
		}

		int indicesCount = ySegments * (xSegments + 1) * 2;

		int* indices = new int[indicesCount];
		index = 0;

		for (int y = 0; y < ySegments; y++)
		{
			if (y % 2 == 0)
				for (int x = 0; x <= xSegments; x++) 
				{
					indices[index++] = y * (xSegments + 1) + x;
					indices[index++] = (y + 1) * (xSegments + 1) + x;
				}
			else
				for (int x = xSegments; x >= 0; x--)
				{
					indices[index++] = (y + 1) * (xSegments + 1) + x;
					indices[index++] = y * (xSegments + 1) + x;
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
		vao->setVertexAttributes(0, {VertexAttrib::Position, VertexAttrib::Normal, VertexAttrib::TexCoords});

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

	Mesh* Primitives::createSphereMesh(int xSegments, int ySegments)
	{
		VertexArray* vao = createSphereVAO(xSegments, ySegments);

		Mesh* mesh = new Mesh(vao);
		mesh->m_IndexBufferOffset = 0;
		mesh->m_IndexType = GL_UNSIGNED_INT;
		mesh->m_IndexCount = vao->indexBuffer()->size() / sizeof(int);
		mesh->m_DrawMode = GL_TRIANGLE_STRIP;

		return mesh;
	}

	VertexArray* Primitives::createCubeVAO()
	{
		VertexArray* vao = new VertexArray();
		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = cubeVertices.size() * sizeof(float);
		vboAllocInfo.data = (void*)cubeVertices.data();
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		vao->addVertexBuffer(0, vbo, 8 * sizeof(float));
		vao->setVertexAttributes(0, {VertexAttrib::Position, VertexAttrib::Normal, VertexAttrib::TexCoords});
		vao->setDestroyBuffersOnDelete();

		return vao;
	}

	VertexArray* Primitives::createQuadVAO()
	{
		VertexArray* vao = new VertexArray();
		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = quadVertices.size() * sizeof(float);
		vboAllocInfo.data = (void*)quadVertices.data();
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		vao->addVertexBuffer(0, vbo, 5 * sizeof(float));
		vao->setVertexAttributes(0, {VertexAttrib::Position, VertexAttrib::TexCoords});
		vao->setDestroyBuffersOnDelete();

		return vao;
	}
}