#pragma once

#include "engine/Engine.h"

using namespace utad;

bool g_WindowActive = true;

struct UIInfo
{
    ArrayList<PostFX>* activeEffects;
    Camera* camera;
};

void drawUI(const UIInfo& info)
{

    ImGui::Begin("Active Post Effects", &g_WindowActive);
    {
        ImGui::SetWindowPos({10, 10});
        ImGui::SetWindowSize({500, 300});
        
        ImGui::Text("Select the effects you want to apply to the scene.");
        ImGui::Text("Effects will be executed in the order you select them.");
        ImGui::Text("Post Effects are accumulative, so order is important!");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        for (size_t i = 0; i < PostFXCount; ++i)
        {
            PostFX fx = static_cast<PostFX>(i);
            auto pos = std::find(info.activeEffects->begin(), info.activeEffects->end(), fx);
            bool selected = pos != info.activeEffects->end();
            bool wasSelected = selected;

            if (ImGui::Checkbox(PostFXNames[i].c_str(), &selected)) // Pressed
            {
                if (!wasSelected && selected)
                    info.activeEffects->push_back(fx);
                else if(wasSelected && !selected)
                    info.activeEffects->erase(pos);
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float exposure = info.camera->exposure();
        ImGui::SliderFloat("Camera Exposure", &exposure, 0, 10.0f);
        info.camera->exposure(exposure);
    }
    ImGui::End();
}