#include "engine/io/Files.h"
#include <fstream>

namespace utad
{

	BinaryFile Files::readAllBytes(const String& path)
	{
		std::ifstream inputStream(path, std::ios::binary | std::ios::ate);

		uint32_t size = static_cast<uint32_t>(inputStream.tellg());

		char* contents = new char[size];
		inputStream.seekg(0, std::ios::beg);
		inputStream.read(contents, size);

		return { contents, size };
	}

	String Files::readAllText(const String& path)
	{
		std::ifstream inputStream(path);
		if (!inputStream.is_open()) throw UTAD_EXCEPTION(String("Failed to open file ").append(path));
		std::stringstream stringStream;
		stringStream << inputStream.rdbuf();
		return stringStream.str();
	}

	bool Files::exists(const String& file)
	{
		return std::filesystem::exists(file);
	}

}