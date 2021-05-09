#pragma once
#include "engine/Common.h"
#include "engine/events/Event.h"
#include "engine/events/KeyboardEvents.h"
#include "engine/events/MouseEvents.h"
#include "engine/events/WindowEvents.h"

namespace utad
{

	class EventSystem
	{
		friend class Engine;
	private:
		static EventSystem* s_Instance;
		Map<EventType, ArrayList<EventCallback>> m_EventCallbacks;
		Queue<Event*> m_EventQueue;
	public:
		static void addEventCallback(EventType type, const EventCallback& callback);
		static void registerEvent(Event* event);
		static void pollEvents();
		static void waitEvents(float timeout = 0.0f);

		~EventSystem() = default;

	private:
		static EventSystem* init();
		static void destroy();
		static void update();

		EventSystem() = default;
	};
}
