#include "engine/events/EventSystem.h"
#include "engine/graphics/GraphicsAPI.h"
#include <iostream>

namespace utad
{
	EventSystem* EventSystem::s_Instance;

	EventSystem* EventSystem::init()
	{
		s_Instance = new EventSystem();
		return s_Instance;
	}

	void EventSystem::destroy()
	{
		UTAD_DELETE(s_Instance);
	}

	void EventSystem::addEventCallback(EventType type, const EventCallback& callback)
	{
		s_Instance->m_EventCallbacks[type].push_back(callback);
	}

	void EventSystem::registerEvent(Event* event)
	{
		if (event != nullptr)
			s_Instance->m_EventQueue.push_back(event);
	}

	void EventSystem::pollEvents()
	{
		glfwPollEvents();
	}

	void EventSystem::waitEvents(float timeout)
	{
		if (timeout == 0.0f)
			glfwWaitEvents();
		else
			glfwWaitEventsTimeout(timeout);
	}

	void EventSystem::update()
	{
		pollEvents();

		Queue<Event*>& eventQueue = s_Instance->m_EventQueue;
		Map<EventType, ArrayList<EventCallback>>& eventCallbacks = s_Instance->m_EventCallbacks;

		size_t eventsCount = eventQueue.size();

		for (size_t i = 0; i < eventsCount; ++i)
		{
			Event* event = eventQueue.front();
			eventQueue.pop_front();

			if (event == nullptr)
			{
				std::cout << "Event is null!!\n";
				continue;
			}

			EventType eventType = event->type();

			if (event->consumed())
			{
				std::cout << "Event is already consumed...\n";
				continue;
			}

			try
			{
				if (eventCallbacks.find(eventType) != eventCallbacks.end())
				{
					ArrayList<EventCallback>& callbacks = eventCallbacks[eventType];
					for (EventCallback& callback : callbacks)
					{
						callback(*event);
						if (event->consumed()) break;
					}
				}
			}
			catch (const std::exception& exception)
			{
				std::cout << String("Error while processing events ").append(exception.what()) << '\n';
			}

			delete event;
		}
	}

}