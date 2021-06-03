#pragma once

#include "GraphicsAPI.h"

namespace utad
{
	void debugCallback(GLuint source, GLuint type, GLuint id, GLuint severity, GLint length,
		const char* message, const void* userParam);
}