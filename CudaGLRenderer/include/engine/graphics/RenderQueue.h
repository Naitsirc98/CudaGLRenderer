#pragma once

#include "engine/graphics/GraphicsAPI.h"
#include "engine/collisions/Collisions.h"

namespace utad
{
	class Mesh;
	class Material;

	const String NO_RENDER_QUEUE = "";
	const String DEFAULT_RENDER_QUEUE = "DEFAULT";

	struct RenderCommand
	{
		String queue;
		Matrix4* transformation;
		Mesh* mesh;
		Material* material;
		AABB* aabb;
	};

	struct RenderQueue
	{
		String name;
		ArrayList<RenderCommand> commands;
		bool enabled{ true };

		RenderQueue()
		{
			commands.reserve(1024);
		}
	};
}