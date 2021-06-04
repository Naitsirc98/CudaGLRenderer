#pragma once

#include "engine/graphics/Texture.h"

namespace utad
{
	struct SkyboxLoadInfo;

	struct Skybox
	{
		friend class SkyboxLoader;

		Cubemap* environmentMap{nullptr};
		Cubemap* irradianceMap{nullptr};
		Cubemap* prefilterMap{nullptr};
		Texture2D* brdfMap{nullptr};
		float maxPrefilterLOD{4.0f};
		float prefilterLODBias{-0.5f};

		~Skybox()
		{
			UTAD_DELETE(environmentMap);
			UTAD_DELETE(irradianceMap);
			UTAD_DELETE(prefilterMap);
			UTAD_DELETE(brdfMap);
		}

	private:
		Skybox() = default;

	public:
		static Skybox* create(const String& path, const SkyboxLoadInfo& loadInfo);
	};
}