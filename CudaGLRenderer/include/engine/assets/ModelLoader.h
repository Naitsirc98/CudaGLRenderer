#pragma once

#include "Mesh.h"

namespace utad
{
	class ModelLoader
	{
	public:
		ModelLoader();
		~ModelLoader();
		Model* load(const String& name, const String& path);
	};

}