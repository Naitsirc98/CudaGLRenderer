#include "engine/Engine.h"

using namespace utad;

static const char* CUBE = "G:/glTF-Sample-Models-master/2.0/Cube/glTF/Cube.gltf";
static const char* BOX = "G:/glTF-Sample-Models-master/2.0/Box/glTF/Box.gltf";
static const char* HELMET = "G:/glTF-Sample-Models-master/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf";

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


class Move : public Script
{
private:
	Transform* m_Transform;
	float m_Direction = 1;
	float m_LastTime = Time::time();
public:
	void onStart() override
	{
		m_Transform = &entity()->transform();
	}

	void onUpdate() override
	{
		m_Transform->position() += Vector3(0, 0, 1) * Time::deltaTime() * m_Direction;

		if (Time::time() - m_LastTime >= 2.5f)
		{
			m_Direction *= -1;
			m_LastTime = Time::time();
		}
	}
};

class MyApp : public Application
{
public:
	void onStart()
	{
		Scene& scene = Scene::get();

		scene.setSkybox(Assets::get().loadSkybox("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/skybox/indoor.hdr"));

		scene.dirLight().direction = {0, 0, 1};
		scene.dirLight().color = {10, 10, 10};
		scene.enableDirLight(true);

		Light light = {};
		light.position = {0, 1, 10};
		light.color = {30, 30, 30};
		scene.pointLights().push_back(std::move(light));

		Entity* cameraController = Entity::create("CameraController");
		cameraController->addScript(new CameraController());

		ModelLoader loader;
		loader.debugMode(true);
		Model* model = loader.load("TheModel", HELMET);

		Entity* object = Entity::create();
		object->meshView().mesh(model->meshes()[0]);
		object->meshView().material(model->materials()[0]);
		object->transform().rotate(45, Vector3(1, 0, 0));
		object->setEnabled(false);

		scene.camera().position({0, 0, 10});
	}

	void onUpdate()
	{
		if (Input::isKeyActive(Key::Key_Space))
		{
			std::cout << "Space key active" << std::endl;
		}
	}
};

int main()
{
	std::cout << "Launching engine..." << std::endl;
	MyApp app;
	return utad::Engine::launch(app);
}
