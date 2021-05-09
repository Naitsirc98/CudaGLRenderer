#include "engine/Engine.h"

using namespace utad;

class MyApp : public Application
{
public:
	void onStart()
	{
		Scene& scene = Scene::get();

		Entity* entity = Entity::create();
	}

	void onUpdate()
	{
		if (Input::isKeyActive(Key::Key_Space))
		{
			int x = 0;
			std::cout << "Hey" << std::endl;
		}
	}
};

int main()
{
	std::cout << "Launching engine..." << std::endl;
	MyApp app;
	return utad::Engine::launch(app);
}
