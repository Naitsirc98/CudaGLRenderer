#include "engine/assets/SkyboxLoader.h"
#include "engine/io/Files.h"
#include "engine/assets/Image.h"

namespace utad
{
	static const GLenum QUAD_DRAW_MODE = GL_TRIANGLE_STRIP;
	static const size_t QUAD_VERTICES_COUNT = 4;
	static float QUAD_VERTICES[] = {
		// positions        // texture Coords
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f
	};

	static const GLenum CUBE_DRAW_MODE = GL_TRIANGLES;
	static const size_t CUBE_VERTICES_COUNT = 36;
	static float CUBE_VERTICES[] = {
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

	SkyboxLoader::SkyboxLoader()
	{
	}

	SkyboxLoader::~SkyboxLoader()
	{
	}

	void SkyboxLoader::init()
	{
		createFramebuffer();
		createVertexData();
		createShaders();
	}

	Skybox* SkyboxLoader::loadSkybox(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo)
	{
		Skybox* skybox = new Skybox();

		skybox->maxPrefilterLOD = loadInfo.maxLOD;
		skybox->prefilterLODBias = loadInfo.lodBias;

		return skybox;
	}

	Cubemap* SkyboxLoader::createEnvironmentMap(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo)
	{
		Texture2D* hdrTexture = loadHDRTexture(hdrImagePath, loadInfo);

		Cubemap* environmentMap = new Cubemap();
		
		TextureAllocInfo allocInfo = {};
		allocInfo.levels = 4;
		allocInfo.format = GL_RGB16F;
		allocInfo.width = loadInfo.environmentMapSize;
		allocInfo.height = loadInfo.environmentMapSize;

		environmentMap->allocate(allocInfo);
		environmentMap->wrap(GL_CLAMP_TO_EDGE);
		environmentMap->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		environmentMap->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		bakeEnvironmentMap(hdrTexture, environmentMap, loadInfo.environmentMapSize);

		return environmentMap;
	}

	Texture2D* SkyboxLoader::loadHDRTexture(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo)
	{
		Texture2D* hdrTexture = new Texture2D();
		Image* image = ImageFactory::createImage(hdrImagePath, GL_RGBA16F, true);

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGBA16F;
		allocInfo.width = image->width();
		allocInfo.height = image->height();
		allocInfo.levels = 0;

		hdrTexture->allocate(std::move(allocInfo));

		hdrTexture->wrap(GL_CLAMP_TO_EDGE);
		hdrTexture->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		hdrTexture->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		UTAD_DELETE(image);

		return hdrTexture;
	}

	void SkyboxLoader::bakeEnvironmentMap(Texture2D* hdrTexture, Cubemap* environmentMap, int size)
	{
		m_EnvironmentMapShader->bind();
		{
			m_EnvironmentMapShader->setTexture("u_EquirectangularMap", hdrTexture);
			renderCubemap(environmentMap, m_EnvironmentMapShader, size, 0);
		}
		m_EnvironmentMapShader->unbind();

		environmentMap->generateMipmaps();
	}

	Cubemap* SkyboxLoader::createIrradianceMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo)
	{
		Cubemap* irradianceMap = new Cubemap();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGB16F;
		allocInfo.width = loadInfo.irradianceMapSize;
		allocInfo.height = loadInfo.irradianceMapSize;
		allocInfo.levels = 4;

		irradianceMap->allocate(std::move(allocInfo));

		irradianceMap->wrap(GL_CLAMP_TO_EDGE);
		irradianceMap->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		irradianceMap->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		irradianceMap->generateMipmaps();

		bakeIrradianceMap(environmentMap, irradianceMap, loadInfo.irradianceMapSize);

		return irradianceMap;
	}

	void SkyboxLoader::bakeIrradianceMap(Cubemap* environmentMap, Cubemap* irradianceMap, int size)
	{
		m_IrradianceShader->bind();
		{
			m_IrradianceShader->setTexture("u_EnvironmentMap", environmentMap);

			renderCubemap(irradianceMap, m_IrradianceShader, size, 0);
		}
		m_IrradianceShader->unbind();

		irradianceMap->generateMipmaps();
	}

	Cubemap* SkyboxLoader::createPrefilterMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo)
	{
		Cubemap* prefilterMap = new Cubemap();

		const int mipLevels = static_cast<int>(roundf(loadInfo.maxLOD) + 1);
		
		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGB16F;
		allocInfo.width = loadInfo.prefilterMapSize;
		allocInfo.height = loadInfo.prefilterMapSize;
		allocInfo.levels = mipLevels;

		prefilterMap->allocate(std::move(allocInfo));

		bakePrefilterMap(environmentMap, prefilterMap, loadInfo.prefilterMapSize, mipLevels);

		return prefilterMap;
	}

	void SkyboxLoader::bakePrefilterMap(Cubemap* environmentMap, Cubemap* prefilterMap, int size, int mipLevels)
	{
		m_PrefilterShader->bind();
		{
			m_PrefilterShader->setTexture("u_EnvironmentMap", environmentMap);
			m_PrefilterShader->setUniform("u_Resolution", environmentMap->width()); // TODO

			const int minMipLevel = mipLevels - 1;

			for (int mipLevel = 0; mipLevel <= minMipLevel; ++mipLevel)
			{
				const int mipLevelSize = static_cast<int>(size * powf(0.5f, mipLevel));
				const float roughness = (float)mipLevel / (float)minMipLevel;
				m_PrefilterShader->setUniform("u_Roughness", roughness);
				renderCubemap(prefilterMap, m_PrefilterShader, size, mipLevel);
			}
		}
		m_PrefilterShader->unbind();
	}

	Texture2D* SkyboxLoader::createBRDFMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo)
	{
		Texture2D* brdf = new Texture2D();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RG16F;
		allocInfo.width = loadInfo.brdfSize;
		allocInfo.height = loadInfo.brdfSize;
		allocInfo.levels = 1;

		brdf->allocate(std::move(allocInfo));

		brdf->wrap(GL_CLAMP_TO_EDGE);
		brdf->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		brdf->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		bakeBRDFMap(brdf, loadInfo.brdfSize);

		return brdf;
	}

	void SkyboxLoader::bakeBRDFMap(Texture2D* brdfMap, int size)
	{
		m_Framebuffer->addTextureAttachment(GL_COLOR_ATTACHMENT0, brdfMap, 0);
		m_Framebuffer->ensureComplete();

		m_Framebuffer->bind();
		{
			glDisable(GL_DEPTH_TEST);
			glViewport(0, 0, size, size);
			glClear(GL_COLOR_BUFFER_BIT);

			m_BRDFShader->bind();
			{
				m_QuadVAO->bind();
				{
					glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
				}
				m_QuadVAO->unbind();
			}
			m_BRDFShader->unbind();

			m_Framebuffer->detachTextureAttachment(GL_COLOR_ATTACHMENT0);
		}
		m_Framebuffer->unbind();
	}

	static Matrix4 getProjectionMatrix()
	{
		return math::perspective(math::radians(90.0f), 1.0f, 0.1f, 100.0f);
	}

	static Array<Matrix4, 6> getViewMatrices()
	{
		return {
			math::lookAt(Vector3(0.0f), {1.0f, 0.0f, 0.0f},  {0.0f, -1.0f, 0.0f}),
			math::lookAt(Vector3(0.0f), {-1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}),
			math::lookAt(Vector3(0.0f), {0.0f, 1.0f, 0.0f},  {0.0f, 0.0f, 1.0f}),
			math::lookAt(Vector3(0.0f), {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}),
			math::lookAt(Vector3(0.0f), {0.0f, 0.0f, 1.0f},  {0.0f, -1.0f, 0.0f}),
			math::lookAt(Vector3(0.0f), {0.0f, 0.0f, -1.0f}, {0.0f, -1.0f, 0.0f})
		};
	}

	void SkyboxLoader::renderCubemap(Cubemap* cubemap, Shader* shader, int size, int mipmapLevel)
	{
		Matrix4 projMatrix = getProjectionMatrix();
		Array<Matrix4, 6> viewMatrices = getViewMatrices();

		m_CubeVAO->bind();
		{
			m_Framebuffer->bind();
			{
				glDisable(GL_DEPTH_TEST);
				glViewport(0, 0, size, size);

				for (int i = 0; i < 6;++i)
				{
					GLenum face = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
					
					shader->setUniform("u_ProjectionViewMatrix", projMatrix * viewMatrices[i]);
					
					m_Framebuffer->bind();

					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, face, cubemap->handle(), mipmapLevel);

					glDrawArrays(GL_TRIANGLES, 0, 36);
				}
			}
			m_Framebuffer->unbind();
		}
		m_CubeVAO->unbind();
	}

	void SkyboxLoader::createFramebuffer()
	{
		m_Framebuffer = new Framebuffer();
	}

	void SkyboxLoader::createVertexData()
	{
		createQuad();
		createCube();
	}

	void SkyboxLoader::createQuad()
	{
		m_QuadVAO = new VertexArray();

		VertexBuffer* vbo = new VertexBuffer();
		
		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = QUAD_VERTICES_COUNT * sizeof(float);
		vboAllocInfo.data = QUAD_VERTICES;
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		VertexAttrib position = {};
		position.count = 3;
		position.type = GL_FLOAT;

		VertexAttrib texCoords = {};
		texCoords.count = 2;
		texCoords.type = GL_FLOAT;

		m_QuadVAO->addVertexBuffer(0, vbo, 5 * sizeof(float));
		m_QuadVAO->setVertexAttrib(0, position, 0, 0);
		m_QuadVAO->setVertexAttrib(0, texCoords, 1, 3 * sizeof(float));

		m_QuadVAO->setDestroyBuffersOnDelete();
	}

	void SkyboxLoader::createCube()
	{
		m_CubeVAO = new VertexArray();

		VertexBuffer* vbo = new VertexBuffer();

		BufferAllocInfo vboAllocInfo = {};
		vboAllocInfo.size = QUAD_VERTICES_COUNT * sizeof(float);
		vboAllocInfo.data = QUAD_VERTICES;
		vboAllocInfo.storageFlags = GPU_STORAGE_LOCAL_FLAGS;

		vbo->allocate(std::move(vboAllocInfo));

		VertexAttrib position = {};
		position.count = 3;
		position.type = GL_FLOAT;

		VertexAttrib normal = {};
		normal.count = 3;
		normal.type = GL_FLOAT;

		VertexAttrib texCoords = {};
		texCoords.count = 2;
		texCoords.type = GL_FLOAT;

		m_CubeVAO->addVertexBuffer(0, vbo, 8 * sizeof(float));
		m_CubeVAO->setVertexAttrib(0, position, 0, 0);
		m_CubeVAO->setVertexAttrib(0, normal, 1, 3 * sizeof(float));
		m_CubeVAO->setVertexAttrib(0, texCoords, 2, 6 * sizeof(float));

		m_CubeVAO->setDestroyBuffersOnDelete();
	}

	void SkyboxLoader::createShaders()
	{
		createEnvironmentMapShader();
		createIrradianceMapShader();
		createPrefilterMapShader();
		createBRDFMapShader();
	}

	static const String SHADERS_DIR = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/skybox/";

	static const char* ENVIRONMENT_MAP_VERTEX_SHADER_FILE = "equirect_to_cubemap.vert";
	static const char* ENVIRONMENT_MAP_FRAGMENT_SHADER_FILE = "equirect_to_cubemap.frag";

	void SkyboxLoader::createEnvironmentMapShader()
	{
		Shader* shader = new Shader("EnvironmentMap");
		
		ShaderStage vertex = {};
		vertex.name = "EnvironmentMap Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + ENVIRONMENT_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Environment Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + ENVIRONMENT_MAP_FRAGMENT_SHADER_FILE));

		shader->attach(&vertex);
		shader->attach(&frag);

		shader->compile();
	}

	static const char* IRRADIANCE_MAP_VERTEX_SHADER_FILE = "irradiance_map.vert";
	static const char* IRRADIANCE_MAP_FRAGMENT_SHADER_FILE = "irradiance_map.frag";

	void SkyboxLoader::createIrradianceMapShader()
	{
		Shader* shader = new Shader("Irradiance");

		ShaderStage vertex = {};
		vertex.name = "Irradiance Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + IRRADIANCE_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Irradiance Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + IRRADIANCE_MAP_FRAGMENT_SHADER_FILE));

		shader->attach(&vertex);
		shader->attach(&frag);

		shader->compile();
	}

	static const char* PREFILTER_MAP_VERTEX_SHADER_FILE = "prefilter_map.vert";
	static const char* PREFILTER_MAP_FRAGMENT_SHADER_FILE = "prefilter_map.frag";

	void SkyboxLoader::createPrefilterMapShader()
	{
		Shader* shader = new Shader("PrefilterMap");

		ShaderStage vertex = {};
		vertex.name = "Prefilter Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + PREFILTER_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Prefilter Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + PREFILTER_MAP_FRAGMENT_SHADER_FILE));

		shader->attach(&vertex);
		shader->attach(&frag);

		shader->compile();
	}

	static const char* BRDF_MAP_VERTEX_SHADER_FILE = "brdf.vert";
	static const char* BRDF_MAP_FRAGMENT_SHADER_FILE = "brdf.frag";

	void SkyboxLoader::createBRDFMapShader()
	{
		Shader* shader = new Shader("BRDFMap");

		ShaderStage vertex = {};
		vertex.name = "BRDF Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + BRDF_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "BRDF Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + BRDF_MAP_FRAGMENT_SHADER_FILE));

		shader->attach(&vertex);
		shader->attach(&frag);

		shader->compile();
	}
}