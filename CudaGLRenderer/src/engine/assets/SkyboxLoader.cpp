#include "engine/assets/SkyboxLoader.h"
#include "engine/io/Files.h"
#include "engine/assets/Image.h"
#include "engine/assets/Primitives.h"
#include "engine/graphics/Window.h"

namespace utad
{
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

		skybox->environmentMap = createEnvironmentMap(hdrImagePath, loadInfo);
		skybox->irradianceMap = createIrradianceMap(skybox->environmentMap, loadInfo);
		skybox->prefilterMap = createPrefilterMap(skybox->environmentMap, loadInfo);
		skybox->brdfMap = createBRDFMap(skybox->environmentMap, loadInfo);
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

		UTAD_DELETE(hdrTexture);

		return environmentMap;
	}

	Texture2D* SkyboxLoader::loadHDRTexture(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo)
	{
		Texture2D* hdrTexture = new Texture2D();
		Image* image = ImageFactory::createImage(hdrImagePath, GL_RGB16F, true);
		
		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGB16F;
		allocInfo.width = image->width();
		allocInfo.height = image->height();
		allocInfo.levels = 1;
		
		hdrTexture->allocate(std::move(allocInfo));
		
		Texture2DUpdateInfo updateInfo = {};
		updateInfo.format = GL_RGB;
		updateInfo.level = 0;
		updateInfo.type = GL_FLOAT;
		updateInfo.pixels = image->pixels();

		hdrTexture->update(std::move(updateInfo));
		
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

		bakeIrradianceMap(environmentMap, irradianceMap, loadInfo.irradianceMapSize);

		irradianceMap->wrap(GL_CLAMP_TO_EDGE);
		irradianceMap->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		irradianceMap->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		irradianceMap->generateMipmaps();

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

		prefilterMap->wrap(GL_CLAMP_TO_EDGE);
		prefilterMap->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		prefilterMap->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		prefilterMap->generateMipmaps();

		return prefilterMap;
	}

	void SkyboxLoader::bakePrefilterMap(Cubemap* environmentMap, Cubemap* prefilterMap, int size, int mipLevels)
	{
		m_PrefilterShader->bind();
		{
			m_PrefilterShader->setTexture("u_EnvironmentMap", environmentMap);
			m_PrefilterShader->setUniform("u_Resolution", environmentMap->width());

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

		bakeBRDFMap(brdf, loadInfo.brdfSize);

		brdf->wrap(GL_CLAMP_TO_EDGE);
		brdf->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		brdf->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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
					glDrawArrays(Primitives::quadDrawMode, 0, Primitives::quadVertexCount);
				}
				m_QuadVAO->unbind();
			}
			m_BRDFShader->unbind();
		}
		m_Framebuffer->unbind();

		m_Framebuffer->detachTextureAttachment(GL_COLOR_ATTACHMENT0);
	}

	static Matrix4 getProjectionMatrix()
	{
		return math::perspective(math::radians(90.0f), 1.0f, 0.1f, 10.0f);
	}

	static Array<Matrix4, 6> getViewMatrices()
	{
		return {
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f,  0.0f,  0.0f), Vector3(0.0f, -1.0f,  0.0f)),
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(-1.0f,  0.0f,  0.0f),Vector3(0.0f, -1.0f,  0.0f)),
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f,  1.0f,  0.0f), Vector3(0.0f,  0.0f,  1.0f)),
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, -1.0f,  0.0f), Vector3(0.0f,  0.0f, -1.0f)),
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f,  0.0f,  1.0f), Vector3(0.0f, -1.0f,  0.0f)),
			math::lookAt(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f,  0.0f, -1.0f), Vector3(0.0f, -1.0f,  0.0f))
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

					glClear(GL_COLOR_BUFFER_BIT);

					glDrawArrays(Primitives::cubeDrawMode, 0, Primitives::cubeVertexCount);
				}
			}
			m_Framebuffer->unbind();
		}
		m_CubeVAO->unbind();

		glViewport(0, 0, Window::get().width(), Window::get().height());
		glFinish();
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
		m_QuadVAO = Primitives::createQuadVAO();
	}

	void SkyboxLoader::createCube()
	{
		m_CubeVAO = Primitives::createCubeVAO();
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
		m_EnvironmentMapShader = new Shader("EnvironmentMap");
		
		ShaderStage vertex = {};
		vertex.name = "EnvironmentMap Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + ENVIRONMENT_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Environment Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + ENVIRONMENT_MAP_FRAGMENT_SHADER_FILE));

		m_EnvironmentMapShader->attach(&vertex);
		m_EnvironmentMapShader->attach(&frag);

		m_EnvironmentMapShader->compile();
	}

	static const char* IRRADIANCE_MAP_VERTEX_SHADER_FILE = "irradiance_map.vert";
	static const char* IRRADIANCE_MAP_FRAGMENT_SHADER_FILE = "irradiance_map.frag";

	void SkyboxLoader::createIrradianceMapShader()
	{
		m_IrradianceShader = new Shader("Irradiance");

		ShaderStage vertex = {};
		vertex.name = "Irradiance Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + IRRADIANCE_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Irradiance Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + IRRADIANCE_MAP_FRAGMENT_SHADER_FILE));

		m_IrradianceShader->attach(&vertex);
		m_IrradianceShader->attach(&frag);

		m_IrradianceShader->compile();
	}

	static const char* PREFILTER_MAP_VERTEX_SHADER_FILE = "prefilter_map.vert";
	static const char* PREFILTER_MAP_FRAGMENT_SHADER_FILE = "prefilter_map.frag";

	void SkyboxLoader::createPrefilterMapShader()
	{
		m_PrefilterShader = new Shader("PrefilterMap");

		ShaderStage vertex = {};
		vertex.name = "Prefilter Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + PREFILTER_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "Prefilter Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + PREFILTER_MAP_FRAGMENT_SHADER_FILE));

		m_PrefilterShader->attach(&vertex);
		m_PrefilterShader->attach(&frag);

		m_PrefilterShader->compile();
	}

	static const char* BRDF_MAP_VERTEX_SHADER_FILE = "brdf.vert";
	static const char* BRDF_MAP_FRAGMENT_SHADER_FILE = "brdf.frag";

	void SkyboxLoader::createBRDFMapShader()
	{
		m_BRDFShader = new Shader("BRDFMap");

		ShaderStage vertex = {};
		vertex.name = "BRDF Vertex Shader";
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = std::move(Files::readAllText(SHADERS_DIR + BRDF_MAP_VERTEX_SHADER_FILE));

		ShaderStage frag = {};
		frag.name = "BRDF Fragment Shader";
		frag.type = GL_FRAGMENT_SHADER;
		frag.sourceCode = std::move(Files::readAllText(SHADERS_DIR + BRDF_MAP_FRAGMENT_SHADER_FILE));

		m_BRDFShader->attach(&vertex);
		m_BRDFShader->attach(&frag);

		m_BRDFShader->compile();
	}
}