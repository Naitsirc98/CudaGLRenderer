#pragma once

#include "engine/entities/EntityPool.h"
#include "Camera.h"
#include "Light.h"
#include "engine/assets/Skybox.h"
#include "engine/graphics/MeshRenderer.h"
#include "engine/graphics/SkyboxRenderer.h"
#include "engine/graphics/postfx/PostFXRenderer.h"

namespace utad
{
	struct RenderInfo
	{
		friend class Scene;

		Camera camera;
		Light dirLight;
		bool enableDirLight;
		ArrayList<Light> pointLights;
		Skybox* skybox{nullptr};
		ArrayList<PostFX> postEffects;

	private:
		RenderInfo() {}
	};

	class Scene
	{
		friend class Engine;
	private:
		static Scene* s_Instance;
	private:
		static Scene* init();
		static void destroy();
	public:
		static Scene& get();
	private:
		EntityPool* m_EntityPool;
		MeshRenderer* m_MeshRenderer;
		SkyboxRenderer* m_SkyboxRenderer;
		PostFXRenderer* m_PostFXRenderer;
		RenderInfo m_RenderInfo;
	private:
		Scene();
		~Scene();
	public:
		Entity* createEntity(const String& name = "");
		void destroyEntity(Entity* entity);
		void destroyAllEntities();
		Entity* find(const String& name) const;
		Camera& camera();
		Light& dirLight();
		void enableDirLight(bool enable);
		ArrayList<Light>& pointLights();
		Skybox* skybox() const;
		void setSkybox(Skybox* skybox, bool deleteExisting = true);
		ArrayList<PostFX>& postEffects();
	private:
		void update();
		void lastUpdate();
		void render();
	};
}