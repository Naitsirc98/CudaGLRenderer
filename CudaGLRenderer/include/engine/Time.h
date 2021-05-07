#pragma once

namespace utad
{
	class Time
	{
		friend class Engine;
	private:
		static float s_DeltaTime;
		static unsigned int s_Frame;
	public:
		static float time() noexcept;
		static float deltaTime() noexcept;
		static float frame() noexcept;
	};
}