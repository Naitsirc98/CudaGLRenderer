#pragma once

#include "Skybox.h"
#include "engine/graphics/Graphics.h"

namespace utad
{
    struct SkyboxLoadInfo
    {
        int environmentMapSize{2048};
        int irradianceMapSize{64};
        int prefilterMapSize{2048};
        int brdfSize{512};
        float maxLOD{4.0f};
        float lodBias{-0.5f};
    };

    const SkyboxLoadInfo DEFAULT_SKYBOX_LOAD_INFO = {};

	class SkyboxLoader
	{
		friend class AssetsManager;
	private:
        Framebuffer* m_Framebuffer;
        
        Shader* m_EnvironmentMapShader;
        Shader* m_IrradianceShader;
        Shader* m_PrefilterShader;
        Shader* m_BRDFShader;

        VertexArray* m_QuadVAO;
        VertexArray* m_CubeVAO;

	private:
		SkyboxLoader();
		~SkyboxLoader();
        void init();
    public:
        Skybox* loadSkybox(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo = DEFAULT_SKYBOX_LOAD_INFO);
    private:
        Cubemap* createEnvironmentMap(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo);
        Texture2D* loadHDRTexture(const String& hdrImagePath, const SkyboxLoadInfo& loadInfo);
        void bakeEnvironmentMap(Texture2D* hdrTexture, Cubemap* environmentMap, int size);

        Cubemap* createIrradianceMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo);
        void bakeIrradianceMap(Cubemap* environmentMap, Cubemap* irradianceMap, int size);

        Cubemap* createPrefilterMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo);
        void bakePrefilterMap(Cubemap* environmentMap, Cubemap* prefilterMap, int size, int mipLevels);

        Texture2D* createBRDFMap(Cubemap* environmentMap, const SkyboxLoadInfo& loadInfo);
        void bakeBRDFMap(Texture2D* brdfMap, int size);

        void renderCubemap(Cubemap* cubemap, Shader* shader, int size, int mipmapLevel);

        void createFramebuffer();
        void createVertexData();
        void createQuad();
        void createCube();
        void createShaders();
        void createEnvironmentMapShader();
        void createIrradianceMapShader();
        void createPrefilterMapShader();
        void createBRDFMapShader();
	};


}