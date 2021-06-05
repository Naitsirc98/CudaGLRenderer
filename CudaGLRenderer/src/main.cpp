#include "engine/Engine.h"

using namespace utad;

static const char* CUBE = "G:/glTF-Sample-Models-master/2.0/Cube/glTF/Cube.gltf";
static const char* BOX = "G:/glTF-Sample-Models-master/2.0/Box/glTF/Box.gltf";
static const char* SPHERE = "G:/glTF-Sample-Models-master/2.0/Sphere/Sphere.gltf";
static const char* HELMET = "G:/glTF-Sample-Models-master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";

static const String TEXTURES_DIR = "G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/textures/";

class CameraController : public Script
{
public:

	float normalSpeed = 3;
	Camera* camera;

	void onStart() override
	{
		camera = &Scene::get().camera();
		camera->sensitivity(0.08f);
	}

	void checkWindowSize()
	{
		if(Input::isKeyTyped(Key::Key_F1)) Window::get().state(WindowState::WINDOWED);
		if(Input::isKeyTyped(Key::Key_F2)) Window::get().state(WindowState::MAXIMIZED);
		if(Input::isKeyTyped(Key::Key_F3)) Window::get().state(WindowState::FULLSCREEN);
	}

	void checkGameSettings()
	{
		if(Input::isKeyTyped(Key::Key_Escape))
		{
			Window& window = Window::get();
			if(window.cursorMode() == CursorMode::CAPTURED)
				window.cursorMode(CursorMode::NORMAL);
			else
				window.cursorMode(CursorMode::CAPTURED);
		}

		checkWindowSize();
	}

	void onUpdate() override
	{
		checkGameSettings();

		float speed = this->normalSpeed * Time::deltaTime();
		if(Input::isKeyActive(Key::Key_Left_Shift)) speed *= 2;
		if(Input::isKeyActive(Key::Key_Left_Alt)) speed /= 2;

		if(Input::isKeyActive(Key::Key_W)) camera->move(CameraDirection::Forward, speed);
		if(Input::isKeyActive(Key::Key_S)) camera->move(CameraDirection::Backwards, speed);
		if(Input::isKeyActive(Key::Key_A)) camera->move(CameraDirection::Left, speed);
		if(Input::isKeyActive(Key::Key_D)) camera->move(CameraDirection::Right, speed);
		if(Input::isKeyActive(Key::Key_Space)) camera->move(CameraDirection::Up, speed);
		if(Input::isKeyActive(Key::Key_Left_Control)) camera->move(CameraDirection::Down, speed);

		if (Input::isKeyActive(Key::Key_P))
		{
			std::cout << "Camera position = "
				<< camera->position().x << ", " 
				<< camera->position().y << ", " 
				<< camera->position().z << std::endl;

			std::cout << "Camera direction = "
				<< camera->forward().x << ", "
				<< camera->forward().y << ", "
				<< camera->forward().z << std::endl;
		}

		if(Window::get().cursorMode() == CursorMode::CAPTURED)
		{
			camera->lookAt(Input::getMousePosition());
			camera->zoom(Input::getMouseScroll().y);
		}
	}
};

class MyApp : public Application
{
public:
	Material* createMaterial(const String& name)
	{
		static const float flipY = false;

		Material* mat = Assets::createMaterial(name);
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
		static Mesh* mesh = Primitives::createSphereMesh(64, 64);
		Material* material = createMaterial(name);

		Entity* sphere = Entity::create();
		sphere->transform().position() = pos;
		sphere->meshView().mesh(mesh);
		sphere->meshView().material(material);
		sphere->setEnabled(true);
	}

	void onStart()
	{
		Scene& scene = Scene::get();

		scene.setSkybox(Assets::get().loadSkybox("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/skybox/indoor.hdr"));

		scene.dirLight().direction = {0, 0, 1};
		scene.dirLight().color = {1, 1, 1};
		scene.enableDirLight(true);

		Light light = {};
		light.position = {0, 1, 3};
		light.color = {10, 10, 10};
		scene.pointLights().push_back(std::move(light));

		Entity* cameraController = Entity::create("CameraController");
		cameraController->addScript(new CameraController());

		ModelLoader loader;
		loader.debugMode(true);
		Model* model = loader.load("TheModel", HELMET);

		createSphere("rusted_iron", {0, 0, 0});
		createSphere("gold", {-2.5f, 0, 0});
		createSphere("wall", {2.5f, 0, 0});
		createSphere("plastic", {-5.0f, 0, 0});
		createSphere("grass", {5.0f, 0, 0});

		scene.camera().position({0, 0.5f, 7});
	}

	float lastTime = 0;

	void onUpdate()
	{
		if (Input::isKeyActive(Key::Key_F10) && Time::time() - lastTime >= 0.3f)
		{
			lastTime = Time::time();

			if (Scene::get().postEffects().empty())
			{
				Scene::get().postEffects().push_back(PostFX::Inversion);
			}
			else
			{
				Scene::get().postEffects().clear();
			}
		}
	}
};

int main()
{
	std::cout << "Launching engine..." << std::endl;
	MyApp app;
	return utad::Engine::launch(app);
}
