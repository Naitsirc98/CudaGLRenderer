#include "engine/graphics/GraphicsAPI.h"

namespace utad
{
	static const char* sourceAsString(int source)
	{
		switch (source) {
		case GL_DEBUG_SOURCE_API:
			return "API";
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
			return "WINDOW SYSTEM";
		case GL_DEBUG_SOURCE_SHADER_COMPILER:
			return "SHADER COMPILER";
		case GL_DEBUG_SOURCE_THIRD_PARTY:
			return "THIRD PARTY";
		case GL_DEBUG_SOURCE_APPLICATION:
			return "APPLICATION";
		case GL_DEBUG_SOURCE_OTHER:
			return "OTHER";
		default:
			return "UNKNOWN";
		}
	}

	static const char* typeAsString(int type)
	{
		switch (type) {
		case GL_DEBUG_TYPE_ERROR:
			return "ERROR";
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			return "DEPRECATED BEHAVIOR";
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			return "UNDEFINED BEHAVIOR";
		case GL_DEBUG_TYPE_PORTABILITY:
			return "PORTABILITY";
		case GL_DEBUG_TYPE_PERFORMANCE:
			return "PERFORMANCE";
		case GL_DEBUG_TYPE_OTHER:
			return "OTHER";
		case GL_DEBUG_TYPE_MARKER:
			return "MARKER";
		default:
			return "UNKNOWN";
		}
	}

	static String levelAsString(int severity)
	{
		switch (severity) {
		case GL_DEBUG_SEVERITY_HIGH:
			return "Error";
		case GL_DEBUG_SEVERITY_MEDIUM:
		case GL_DEBUG_SEVERITY_LOW:
			return "Warning";
		default:
			return "Info";
		}
	}

	void debugCallback(GLuint source, GLuint type, GLuint id, GLuint severity, GLint length,
		const char* message, const void* userParam)
	{
		String level = levelAsString(severity);

		if (level == "Info") return;

		String messageStr = String("[OPENGL][")
			.append(std::to_string(id))
			.append("|")
			.append(sourceAsString(source))
			.append("|")
			.append(typeAsString(type))
			.append("]: ")
			.append(message);

		std::cout << messageStr << std::endl;

		if(level == "Error") __debugbreak();
	}
}