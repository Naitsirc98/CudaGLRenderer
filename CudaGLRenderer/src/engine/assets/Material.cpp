#include "engine/assets/Material.h"

namespace utad
{
	Material::Material(MaterialID id) : m_ID(id)
	{
	}

	Material::~Material()
	{
	}

	MaterialID Material::id() const
	{
		return m_ID;
	}

	const Color& Material::albedo() const
	{
		return m_Albedo;
	}

	Material& Material::albedo(const Color& color)
	{
		m_Albedo = color;
		return *this;
	}

	const Color& Material::emissiveColor() const
	{
		return m_EmissiveColor;
	}

	Material& Material::emissiveColor(const Color& color)
	{
		m_EmissiveColor = color;
		return *this;
	}

	Texture2D* Material::albedoMap() const
	{
		return m_AlbedoMap;
	}

	Material& Material::albedoMap(Texture2D* map)
	{
		m_AlbedoMap = map;
		return *this;
	}

	Texture2D* Material::metallicRoughnessMap() const
	{
		return m_MetallicRoughnessMap;
	}

	Material& Material::metallicRoughnessMap(Texture2D* map)
	{
		m_MetallicRoughnessMap = map;
		return *this;
	}

	Texture2D* Material::occlussionMap() const
	{
		return m_OcclussionMap;
	}

	Material& Material::occlussionMap(Texture2D* map)
	{
		m_OcclussionMap = map;
		return *this;
	}

	Texture2D* Material::emissiveMap() const
	{
		return m_EmissiveMap;
	}

	Material& Material::emissiveMap(Texture2D* map)
	{
		m_EmissiveMap = map;
		return *this;
	}

	Texture2D* Material::normalMap() const
	{
		return m_NormalMap;
	}

	Material& Material::normalMap(Texture2D* map)
	{
		m_NormalMap = map;
		return *this;
	}
}