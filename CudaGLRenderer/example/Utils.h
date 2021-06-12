#pragma once

#include "engine/Engine.h"

namespace utad
{
	const char* CUBE = "G:/glTF-Sample-Models-master/2.0/Cube/glTF/Cube.gltf";
	const char* BOX = "G:/glTF-Sample-Models-master/2.0/Box/glTF/Box.gltf";
	const char* SPHERE = "G:/glTF-Sample-Models-master/2.0/Sphere/Sphere.gltf";
	const char* HELMET = "G:/glTF-Sample-Models-master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";

	const String TEXTURES_DIR = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/textures/";
	const String SKYBOX_DIR = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/skybox/";

	const String SKYBOX_INDOOR = SKYBOX_DIR + "indoor.hdr";
	const String SKYBOX_SUNRISE_BEACH = SKYBOX_DIR + "sunrise_beach_2k.hdr";
	const String SKYBOX_NIGHT = SKYBOX_DIR + "night_2k.hdr";

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