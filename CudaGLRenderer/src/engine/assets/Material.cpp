#include "engine/assets/Material.h"
#include "engine/assets/AssetsManager.h"

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

	float Material::metallic() const
	{
		return m_Metallic;
	}

	Material& Material::metallic(float metallic)
	{
		m_Metallic = metallic;
		return *this;
	}

	float Material::roughness() const
	{
		return m_Roughness;
	}

	Material& Material::roughness(float roughness)
	{
		m_Roughness = roughness;
		return *this;
	}

	Texture2D* Material::metallicMap() const
	{
		return m_MetallicMap;
	}

	Material& Material::metallicMap(Texture2D* map)
	{
		m_MetallicMap = map;
		return *this;
	}

	Texture2D* Material::roughnessMap() const
	{
		return m_RoughnessMap;
	}

	Material& Material::roughnessMap(Texture2D* map)
	{
		m_RoughnessMap = map;
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

	float Material::occlusion() const
	{
		return m_Occlusion;
	}

	Material& Material::occlusion(float occlusion)
	{
		m_Occlusion = occlusion;
		return *this;
	}

	Texture2D* Material::occlusionMap() const
	{
		return m_OcclussionMap;
	}

	Material& Material::occlusionMap(Texture2D* map)
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

	float Material::normalScale() const
	{
		return m_NormalScale;
	}

	Material& Material::normalScale(float normalScale)
	{
		m_NormalScale = normalScale;
		return *this;
	}

	Texture2D* Material::normalMap() const
	{
		return m_NormalMap;
	}

	Material& Material::normalMap(Texture2D* map)
	{
		m_NormalMap = map;
		useNormalMap(map != nullptr);
		return *this;
	}

	float Material::fresnel0() const
	{
		return m_Fresnel0;
	}

	Material& Material::fresnel0(float fresnel0)
	{
		m_Fresnel0 = fresnel0;
		return *this;
	}

	float Material::alpha() const
	{
		return m_Alpha;
	}

	Material& Material::alpha(float alpha)
	{
		m_Alpha = alpha;
		return *this;
	}

	bool Material::useNormalMap() const
	{
		return m_UseNormalMap;
	}

	Material& Material::useNormalMap(bool use)
	{
		m_UseNormalMap = use;
		return *this;
	}

	bool Material::useCombinedMetallicRoughnessMap() const
	{
		return m_UseCombinedMetallicRoughnessMap;
	}

	Material& Material::useCombinedMetallicRoughnessMap(bool use)
	{
		m_UseCombinedMetallicRoughnessMap = use;
		return *this;
	}
}