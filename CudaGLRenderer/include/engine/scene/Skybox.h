#pragma once

#include "engine/graphics/Texture.h"

namespace utad
{
	class Skybox
	{
	private:
		Cubemap* u_IrradianceMap;
		Cubemap* u_PrefilterMap;
		Texture2D* u_BRDF;
		float u_MaxPrefilterLOD;
		float u_PrefilterLODBias;
	};
}