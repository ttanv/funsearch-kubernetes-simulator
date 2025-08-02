import heapq
from dataclasses import dataclass
from enum import StrEnum, auto

from simulator.entities import Pod

class EventType(StrEnum):
    CREATION = auto()
    DELETION = auto()

@dataclass
class Event:
    event_type: EventType
    pod: Pod
    
    def __lt__(self, other):
        return self.pod.pod_id < other.pod.pod_id

class DiscreteEventSimulator:
    """A discrete event simulator implementation, sorts events based on time.
       Gets the next event in order, and allows to add new events
    """
    def __init__(self, pod_list: list[Pod]):
        self.event_heap = self._populate_heap(pod_list)
        heapq.heapify(self.event_heap)
        return

    def _populate_heap(self, pod_list: list[Pod]):
        """Populates the heap with creation times. Deletion time depends on scheduler"""
        events = []
        for pod in pod_list:
            event = Event(EventType.CREATION, pod)
            events.append((pod.creation_time, event))
        return events

    def pop_event(self):
        return heapq.heappop(self.event_heap)

    def peak_event(self):
        return self.event_heap[0]
    
    def finished_events(self):
        return len(self.event_heap) == 0

    def push_deletion_event(self, pod):
        del_event = Event(EventType.DELETION, pod)
        # Assumes pod.deletion_time has been set by the scheduler.
        heapq.heappush(self.event_heap, (pod.deletion_time, del_event))
        
    def repush_creation_event(self, pod):
        # In case no resources free
        for (time, event) in self.event_heap:
                if event.event_type == EventType.DELETION:
                    new_time = time + 1
                    new_event = (new_time, Event(EventType.CREATION, pod))
                    heapq.heappush(self.event_heap, new_event)