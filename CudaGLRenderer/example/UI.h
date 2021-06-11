#pragma once

#include "engine/Engine.h"

using namespace utad;

bool g_WindowActive = true;

void drawUI(ArrayList<PostFX>& activeEffects)
{
    ImGui::Begin("Active Post Effects", &g_WindowActive);
    {
        ImGui::Text("Select the effects you want to apply to the scene.");
        ImGui::Text("Effects will be executed in the order you select them.");
        ImGui::Text("Post Effects are accumulative, so order is important!");

        ImGui::Spacing();

        for (size_t i = 0; i < PostFXCount; ++i)
        {
            PostFX fx = static_cast<PostFX>(i);
            auto pos = std::find(activeEffects.begin(), activeEffects.end(), fx);
            bool selected = pos != activeEffects.end();
            bool wasSelected = selected;

            if (ImGui::Checkbox(PostFXNames[i].c_str(), &selected)) // Pressed
            {
                if (!wasSelected && selected)
                    activeEffects.push_back(fx);
                else if(wasSelected && !selected)
                    activeEffects.erase(pos);
            }
        }
    }
    ImGui::End();
}