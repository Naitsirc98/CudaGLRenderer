#pragma once

#include "engine/Common.h"
#include <filesystem>

namespace utad
{
	using Path = std::filesystem::path;

	struct BinaryFile
	{
		byte* data;
		size_t size;
	};

	class Files
	{
	public:
		static BinaryFile readAllBytes(const String & filename);
		static String readAllText(const String& filename);
		static bool exists(const String& filename);
	};
}