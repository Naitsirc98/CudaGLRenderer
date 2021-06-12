#include "Utils.h"
#include "CameraController.h"
#include "UI.h"

using namespace utad;

class MyApp : public Application
{
private:
	ArrayList<PostFX> m_PostEffects;

public:
	void onStart()
	{
		Scene& scene = Scene::get();

		scene.setSkybox(Assets::get().loadSkybox(SKYBOX_SUNRISE_BEACH));

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

		Entity* object = Entity::create();
		object->meshView().mesh(model->meshes()[0]);
		object->meshView().material(model->materials()[0]);
		object->transform().rotate(45, {1, 0, 0});
		object->transform().position() = {0, 3, -2};

		//createSphere("rusted_iron", {0, 0, 0});
		//createSphere("gold", {-2.5f, 0, 0});
		//createSphere("wall", {2.5f, 0, 0});
		//createSphere("plastic", {-5.0f, 0, 0});
		//createSphere("grass", {5.0f, 0, 0});

		scene.camera().position({0, 0.5f, 7});

		UIInfo uiInfo = {};
		uiInfo.activeEffects = &scene.postEffects();
		uiInfo.camera = &scene.camera();

		UIDrawer uiDrawer;
		uiDrawer.name = "ui";
		uiDrawer.callback = [=] {drawUI(uiInfo); };

		UIRenderer::get().addUIDrawer(uiDrawer);
	}

	float lastTime = 0;
	Set<PostFX> activeEffects;

	void triggerEffect(PostFX postFX)
	{
		Scene& scene = Scene::get();
		if (activeEffects.find(postFX) != activeEffects.end())
		{
			const auto pos = std::find(scene.postEffects().begin(), scene.postEffects().end(), postFX);
			scene.postEffects().erase(pos);
			activeEffects.erase(postFX);
		}
		else
		{
			scene.postEffects().push_back(postFX);
			activeEffects.insert(postFX);
		}

		lastTime = Time::time();
	}

	void onUpdate()
	{
		if (Time::time() - lastTime < 0.3f) return;

		if (Input::isKeyActive(Key::Key_V)) Window::get().setVsync(!Window::get().vsync());

		//if (Input::isKeyActive(Key::Key_F6)) triggerEffect(PostFX::Grayscale);
		//if (Input::isKeyActive(Key::Key_F7)) triggerEffect(PostFX::Inversion);
		//if (Input::isKeyActive(Key::Key_F8)) triggerEffect(PostFX::GammaCorrection);
		//if (Input::isKeyActive(Key::Key_F9)) triggerEffect(PostFX::Blur);
		//if (Input::isKeyActive(Key::Key_F10)) triggerEffect(PostFX::Sharpening);
		//if (Input::isKeyActive(Key::Key_F11)) triggerEffect(PostFX::EdgeDetection);
		//if (Input::isKeyActive(Key::Key_F3)) triggerEffect(PostFX::Emboss);
	}
};

int main()
{
	std::cout << "Launching engine..." << std::endl;
	MyApp app;
	return utad::Engine::launch(app);
}
