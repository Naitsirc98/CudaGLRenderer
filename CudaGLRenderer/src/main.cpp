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
};

int main()
{
	MyApp app;
	return utad::Engine::launch(app);
}
