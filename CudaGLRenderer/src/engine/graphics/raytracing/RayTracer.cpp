#include "engine/graphics/raytracing/RayTracer.h"
#include "engine/scene/Scene.h"
#include "engine/graphics/Window.h"
#include "engine/io/Files.h"

namespace utad
{
	RayTracer::RayTracer()
	{
		m_ColorBuffer = new Texture2D();
	}

	RayTracer::~RayTracer()
	{
		UTAD_DELETE(m_ColorBuffer);
		//UTAD_DELETE(m_Shader);
	}

	void RayTracer::render(SceneSetup& scene)
	{
		// TODO: render texture to quad
	}

	void RayTracer::rayTracing(SceneSetup& scene)
	{
		const int height = m_ColorBuffer->height();
		const int width = m_ColorBuffer->width();
		float scale = (float)tan(math::radians(51.52f * 0.5f));
		float aspectRatio = (float)width / (float)height;
		const Camera& camera = scene.camera;

		Color* pixels = (Color*)(m_ColorBuffer->pixels());

		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				//calcular un nuevo rayo desde la cámara
				float x = (2 * (j + 0.5f) / (float)width - 1) * aspectRatio * scale;
				float y = (1 - 2 * (i + 0.5f) / (float)height) * scale;
				glm::vec4 dir = glm::vec4(x, y, -1.0f, 0);
				//colocarlo en coordenadas de la cámara
				dir = camera.viewMatrix() * dir;
				dir = math::normalize(dir);
				glm::vec4 orig = camera.viewMatrix() * Vector4(camera.position(), 1.0);
				//creamos el rayo con los datos
				Ray ray;
				ray.direction = dir;
				ray.origin = orig;
				ray.inside = false;
				//y lanzamos el rayo
				pixels[i * width + j] = traceRay(scene, ray, 4);
			}
		}
	}

	Color RayTracer::traceRay(SceneSetup& scene, const Ray& ray, uint depth)
	{
		SortedMap<float, Collision> collisions;
		getCollisions(scene, ray, collisions);

		if (collisions.empty()) return colors::BLACK;

		auto[distance, collision] = *collisions.begin();

		return computeLightColor(scene, nullptr, ray, collision, depth);
	}

	Color RayTracer::computeLightColor(SceneSetup& scene, Entity* entity, const Ray& ray, Collision& collision, uint depth)
	{
		return Color();
	}

	void RayTracer::prepareColorBuffer()
	{
		const Window& window = Window::get();

		if (m_ColorBuffer == nullptr || m_ColorBuffer->width() == window.width() && m_ColorBuffer->height() == window.height()) return;

		UTAD_DELETE(m_ColorBuffer);
	
		m_ColorBuffer = new Texture2D();

		TextureAllocInfo allocInfo = {};
		allocInfo.format = GL_RGBA32F;
		allocInfo.width = window.width();
		allocInfo.height = window.height();
		allocInfo.levels = 1;
	
		m_ColorBuffer->allocate(std::move(allocInfo));

		m_ColorBuffer->wrap(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		m_ColorBuffer->wrap(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		m_ColorBuffer->wrap(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		m_ColorBuffer->filter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_ColorBuffer->filter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		m_ColorBuffer->mapPixels(GL_RGBA32F, GL_FLOAT, window.width() * window.height() * 4 * sizeof(float));
	}

	void RayTracer::getCollisions(SceneSetup& scene, const Ray& ray, SortedMap<float, Collision>& collisions)
	{
		RenderQueue* queue = scene.renderQueues[DEFAULT_RENDER_QUEUE];

		for (RenderCommand& command : queue->commands)
		{
			command.aabb->getCollisions(ray, collisions);
		}
	}

	void RayTracer::createShader()
	{
		m_Shader = new Shader("Quad Shader");

		ShaderStage vertex = {};
		vertex.type = GL_VERTEX_SHADER;
		vertex.sourceCode = Files::readAllText("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/quad.vert");

		ShaderStage fragment = {};
		fragment.type = GL_FRAGMENT_SHADER;
		fragment.sourceCode = Files::readAllText("G:/Visual Studio Cpp/CudaGLRenderer/CudaGLRenderer/res/shaders/quad.frag");

		m_Shader->attach(&vertex);
		m_Shader->attach(&fragment);

		m_Shader->compile();
	}
}

