#pragma once

#include "engine/graphics/Texture.h"

namespace utad
{
	using MaterialID = uint;

	class Material
	{
		friend class AssetsManager;
	private:
		MaterialID m_ID{NULL};
		// Colors
		Color m_Albedo{colors::WHITE};
		Color m_EmissiveColor{colors::BLACK};
		// Textures
		Texture2D* m_AlbedoMap{nullptr};
		Texture2D* m_MetallicMap{nullptr};
		Texture2D* m_RoughnessMap{nullptr};
		Texture2D* m_MetallicRoughnessMap{nullptr};
		Texture2D* m_OcclussionMap{nullptr};
		Texture2D* m_EmissiveMap{nullptr};
		Texture2D* m_NormalMap{nullptr};
		// Values
		float m_Alpha{1.0f};
		float m_Metallic{1.0f};
		float m_Roughness{1.0f};
		float m_Occlusion{1.0f};
		float m_Fresnel0{0.02f};
		float m_NormalScale{1.0f};
		// Flags
		bool m_UseNormalMap{false};
		bool m_UseCombinedMetallicRoughnessMap{true};
	private:
		Material(MaterialID id);
		~Material();
	public:
		MaterialID id() const;
		const Color& albedo() const;
		Material& albedo(const Color& color);
		Texture2D* albedoMap() const;
		Material& albedoMap(Texture2D* map);
		float metallic() const;
		Material& metallic(float metallic);
		float roughness() const;
		Material& roughness(float roughness);
		Texture2D* metallicMap() const;
		Material& metallicMap(Texture2D* map);
		Texture2D* roughnessMap() const;
		Material& roughnessMap(Texture2D* map);
		Texture2D* metallicRoughnessMap() const;
		Material& metallicRoughnessMap(Texture2D* map);
		float occlusion() const;
		Material& occlusion(float occlussion);
		Texture2D* occlusionMap() const;
		Material& occlusionMap(Texture2D* map);
		const Color& emissiveColor() const;
		Material& emissiveColor(const Color& color);
		Texture2D* emissiveMap() const;
		Material& emissiveMap(Texture2D* map);
		float normalScale() const;
		Material& normalScale(float normalScale);
		Texture2D* normalMap() const;
		Material& normalMap(Texture2D* map);
		float fresnel0() const;
		Material& fresnel0(float fresnel0);
		float alpha() const;
		Material& alpha(float alpha);
		bool useNormalMap() const;
		Material& useNormalMap(bool use);
		bool useCombinedMetallicRoughnessMap() const;
		Material& useCombinedMetallicRoughnessMap(bool use);
	};
}