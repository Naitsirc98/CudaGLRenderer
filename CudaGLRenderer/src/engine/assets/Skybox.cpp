#include "engine/assets/Skybox.h"
#include "engine/assets/SkyboxLoader.h"
#include "engine/assets/AssetsManager.h"

namespace utad
{
	Skybox* Skybox::create(const String& path, const SkyboxLoadInfo& loadInfo)
	{
		return Assets::get().loadSkybox(path, loadInfo);
	}
}