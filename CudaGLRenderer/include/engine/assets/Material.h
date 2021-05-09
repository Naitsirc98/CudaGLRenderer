#pragma once

#include "engine/graphics/Texture.h"

namespace utad
{
	using MaterialID = uint;

	class Material
	{
	private:
		MaterialID m_ID{NULL};
		// Colors
		Color m_Albedo{colors::WHITE};
		Color m_EmissiveColor{colors::BLACK};
		// Textures
		Texture2D* m_AlbedoMap;
		Texture2D* m_MetallicMap;
		Texture2D* m_RoughnessMap;
		Texture2D* m_OcclussionMap;
		Texture2D* m_EmissiveMap;
		Texture2D* m_NormalMap;
		// Values
		float m_Alpha{1.0f};
		float m_Metallic{0.0f};
		float m_Roughness{1.0f};
		float m_Occlusion{0.0f};
		float m_Fresnel0{0.0f};
	private:
		Material(MaterialID id);
		~Material();
	public:
		MaterialID id() const;
		const Color& albedo() const;
		Material& albedo(const Color& color);
		const Color& emissiveColor() const;
		Material& emissiveColor(const Color& color);
		Texture2D* albedoMap() const;
		Material& albedoMap(Texture2D* map);
		Texture2D* metallicMap() const;
		Material& metallicMap(Texture2D* map);
		Texture2D* roughnessMap() const;
		Material& roughnessMap(Texture2D* map);
		Texture2D* occlussionMap() const;
		Material& occlussionMap(Texture2D* map);
		Texture2D* emissiveMap() const;
		Material& emissiveMap(Texture2D* map);
		Texture2D* normalMap() const;
		Material& normalMap(Texture2D* map);
	};
}