#pragma once

#include "engine/Engine.h"

namespace utad
{
	static const char* CUBE = "G:/glTF-Sample-Models-master/2.0/Cube/glTF/Cube.gltf";
	static const char* BOX = "G:/glTF-Sample-Models-master/2.0/Box/glTF/Box.gltf";
	static const char* SPHERE = "G:/glTF-Sample-Models-master/2.0/Sphere/Sphere.gltf";
	static const char* HELMET = "G:/glTF-Sample-Models-master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";

	static const String TEXTURES_DIR = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/textures/";

	Material* createMaterial(const String& name)
	{
		static const float flipY = false;

		Material* mat = Assets::getMaterial(name);

		if (mat != nullptr) return mat;

		mat = Assets::createMaterial(name);

		mat->albedoMap(Texture2D::load(TEXTURES_DIR + name + "/albedo.png", GL_RGBA, flipY));
		mat->normalMap(Texture2D::load(TEXTURES_DIR + name + "/normal.png", GL_RGBA, flipY));
		mat->metallicMap(Texture2D::load(TEXTURES_DIR + name + "/metallic.png", GL_RGBA, flipY));
		mat->roughnessMap(Texture2D::load(TEXTURES_DIR + name + "/roughness.png", GL_RGBA, flipY));
		mat->occlusionMap(Texture2D::load(TEXTURES_DIR + name + "/ao.png", GL_RGBA, flipY));
		mat->emissiveColor(colors::BLACK);

		mat->useNormalMap(true);
		mat->useCombinedMetallicRoughnessMap(false);

		return mat;
	}

	void createSphere(const String& name, const Vector3& pos)
	{
		Material* material = createMaterial(name);

		Entity* sphere = Entity::create();
		sphere->transform().position() = pos;
		sphere->meshView().mesh(MeshPrimitives::sphere());
		sphere->meshView().material(material);
		sphere->setEnabled(true);
	}
}