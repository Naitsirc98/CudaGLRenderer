#pragma once

#include "engine/graphics/GraphicsAPI.h"
#include "engine/collisions/Collisions.h"
#include "engine/graphics/Graphics.h"

namespace utad
{
	struct SceneSetup;
	class Entity;

	class RayTracer
	{
		friend class Graphics;
		friend class Scene;
	private:
		Texture2D* m_ColorBuffer;
		Shader* m_Shader;
	private:
		RayTracer();
		~RayTracer();
	public:
		void render(SceneSetup& info);
	private:
		void rayTracing(SceneSetup& info);
		Color traceRay(SceneSetup& scene, const Ray& ray, uint depth);
		Color computeLightColor(SceneSetup& scene, Entity* entity,
			const Ray& ray, Collision& collision, uint depth);
		void prepareColorBuffer();
		void getCollisions(SceneSetup& scene, const Ray& ray, SortedMap<float, Collision>& collisions);
		void createShader();
	};
}