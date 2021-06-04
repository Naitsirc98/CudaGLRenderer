#version 450 core

#define PI 3.1415926536

struct Material 
{
	vec4 albedo;
	vec4 emissiveColor;

	float alpha;
	float metallic;
	float roughness;
	float occlusion;
	float fresnel0;
	float normalScale;
};

uniform sampler2D u_AlbedoMap;
uniform sampler2D u_MetallicRoughnessMap;
uniform sampler2D u_OcclussionMap;
uniform sampler2D u_EmissiveMap;
uniform sampler2D u_NormalMap;

struct Light
{
    vec3 color;

    vec3 position;
    vec3 direction;

    float constant;
    float linear;
    float quadratic;

    float ambientFactor;
};

struct Camera
{
    vec3 position;
};

struct RenderInfo 
{
    vec3 albedo;
    vec3 normal;
    vec3 viewDir;
    vec3 reflectDir;
    vec3 F0;

    float metallic;
    float roughness;
    float occlussion; 
};

uniform Material u_Material;

uniform Camera u_Camera;

uniform Light u_DirLight;
uniform bool u_DirLightPresent;

uniform Light u_PointLights[20];
uniform int u_PointLightsCount;

uniform bool u_SkyboxPresent;
uniform samplerCube u_IrradianceMap;
uniform samplerCube u_PrefilterMap;
uniform sampler2D u_BRDF;
uniform float u_MaxPrefilterLOD;
uniform float u_PrefilterLODBias;

uniform vec3 u_AmbientColor;

layout(location = 0) in Fragment
{
	vec3 position;
	vec3 normal;
	vec2 texCoords;
} fragment;

layout(location = 0) out vec4 out_FragmentColor;

RenderInfo g_Info;

float radicalInverseVanDerCorpus(uint bits)
{   
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint N)
{
	return vec2(float(i) / float(N), radicalInverseVanDerCorpus(i));
}

vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness * roughness;

	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;

	vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);

	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;

	return normalize(sampleVec);
}

float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) 
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 calculateLighting(vec3 lightColor, vec3 L, vec3 H, float attenuation) 
{
    vec3 normal = g_Info.normal;
    vec3 viewDir = g_Info.viewDir;
    vec3 F0 = g_Info.F0;
    vec3 albedo = g_Info.albedo;
    float metallic = g_Info.metallic;
    float roughness = g_Info.roughness;

    vec3 radiance = lightColor * attenuation;

    float NDF = distributionGGX(normal, H, roughness);
    float G = geometrySmith(normal, viewDir, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, normal), 0.0), F0);

    vec3 nominator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, L), 0.0);
    vec3 specular = nominator / (denominator + 0.0001); // 0.001 to prevent division by zero.

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(normal, L), 0.0);

    vec3 L0 = (kD * albedo / PI + specular) * radiance * NdotL;

    return L0;
}

vec3 computeDirLights() 
{
    if(!u_DirLightPresent) return vec3(0.0);

    vec3 L = normalize(u_DirLight.direction.xyz);
    vec3 H = normalize(g_Info.viewDir + L);
    float distance = length(u_DirLight.direction.xyz);
    float attenuation = 1.0; // No attenuation in directional lights

    return calculateLighting(u_DirLight.color.rgb, L, H, attenuation);
}

vec3 computePointLights()
{
    vec3 L0 = vec3(0.0);

    for(int i = 0; i < u_PointLightsCount; ++i) {

        Light light = u_PointLights[i];

        vec3 direction = light.position.xyz - fragment.position;

        vec3 L = normalize(direction);
        vec3 H = normalize(g_Info.viewDir + L);
        float distance = length(direction);
        float attenuation = 1.0 / (distance * distance);

        L0 += calculateLighting(light.color.rgb, L, H, attenuation);
    }

    return L0;
}

vec3 reflectanceEquation()
{
    vec3 dirLighting = computeDirLights();
    vec3 pointLighting = computePointLights();
    return dirLighting + pointLighting;
}

vec3 getDiffuseIBL()
{
    return g_Info.albedo * texture(u_IrradianceMap, g_Info.normal).rgb;
}

vec3 getSpecularIBL(vec3 F, float angle) 
{
    float prefilterLOD = g_Info.roughness * u_MaxPrefilterLOD + u_PrefilterLODBias;
    vec3 prefilteredColor = textureLod(u_PrefilterMap, g_Info.reflectDir, prefilterLOD).rgb;
    vec2 brdf = texture(u_BRDF, vec2(angle, g_Info.roughness)).rg;
    return prefilteredColor * (F * brdf.x + brdf.y);    
}

vec3 getNormalFromMap(Material material, vec2 uv, vec3 position, vec3 normal)
{
    vec3 tangentNormal = texture(u_NormalMap, uv).xyz * 2.0 - 1.0;

    vec3 Q1 = dFdx(position);
    vec3 Q2 = dFdy(position);
    vec2 st1 = dFdx(uv);
    vec2 st2 = dFdy(uv);

    vec3 N = normalize(normal);
    vec3 T = normalize(Q1 * st2.t - Q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal * material.normalScale);
 }

 vec3 getNormal(Material material, vec2 uv, vec3 position, vec3 normal)
 {
     return getNormalFromMap(material, uv, position, normal);
 }

vec4 getAlbedo(Material material, vec2 uv) 
{
    return material.albedo * pow(texture(u_AlbedoMap, uv), vec4(2.2));
}

float getMetallic(Material material, vec2 uv)
{
    return texture(u_MetallicRoughnessMap, uv).r * material.metallic;
}

float getRoughness(Material material, vec2 uv)
{
    return texture(u_MetallicRoughnessMap, uv).g * material.roughness;
}

float getOcclusion(Material material, vec2 uv) 
{
    return texture(u_OcclussionMap, uv).r * material.occlusion;
}

vec3 getF0(Material material, vec3 albedo, float metallic) 
{
    return mix(vec3(material.fresnel0), albedo, metallic);
}

vec4 computeLighting()
{
    vec2 texCoords = fragment.texCoords;

    g_Info.albedo = getAlbedo(u_Material, texCoords).rgb;
    g_Info.metallic = getMetallic(u_Material, texCoords);
    g_Info.roughness = getRoughness(u_Material, texCoords);
    g_Info.occlussion = getOcclusion(u_Material, texCoords);
    g_Info.normal = getNormal(u_Material, texCoords, fragment.position, fragment.normal);
    g_Info.F0 = getF0(u_Material, g_Info.albedo, g_Info.metallic);

    g_Info.viewDir = normalize(u_Camera.position - fragment.position);
    g_Info.reflectDir = reflect(-g_Info.viewDir, g_Info.normal);

    float angle = max(dot(g_Info.normal, g_Info.viewDir), 0.0);

    vec3 L0 = reflectanceEquation();
    vec3 F = fresnelSchlickRoughness(angle, g_Info.F0, g_Info.roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD = kD * (1.0 - g_Info.metallic);

    vec3 ambient;

    if(u_SkyboxPresent)
    {
        // If skybox is present, then apply Image Based Lighting (IBL)
        ambient = (kD * getDiffuseIBL() + getSpecularIBL(F, angle)) * g_Info.occlussion;
    }
    else
    {
        ambient = kD * u_AmbientColor * g_Info.albedo * g_Info.occlussion;
    }

    return vec4(ambient + L0, 1.0);
}

void main()
{
	vec4 color = computeLighting();

    vec4 emissive = u_Material.emissiveColor * texture(u_EmissiveMap, fragment.texCoords);

	out_FragmentColor = vec4((color + emissive).rgb, u_Material.alpha);
}