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
        
        # Used to easily get the next del event time
        self.deletion_heap = []
        return

    def _populate_heap(self, pod_list: list[Pod]):
        """Populates the heap with creation times. Deletion time depends on scheduler"""
        events = []
        for pod in pod_list:
            event = Event(EventType.CREATION, pod)
            events.append((pod.creation_time, event))
        return events

    def pop_event(self):
        popped_event = heapq.heappop(self.event_heap)
        
        # Ensure element is also removed from del heap
        if popped_event[1].event_type == EventType.DELETION:
            heapq.heappop(self.deletion_heap)
        
        return popped_event

    def peak_event(self):
        return self.event_heap[0]
    
    def finished_events(self):
        return len(self.event_heap) == 0

    def push_deletion_event(self, pod: Pod):
        del_event = Event(EventType.DELETION, pod)
        deletion_time = pod.creation_time + pod.duration_time
        
        heapq.heappush(self.deletion_heap, deletion_time)
        heapq.heappush(self.event_heap, (deletion_time, del_event))
        
    def repush_creation_event(self, pod: Pod):
        # In case no resources free
        post_deletion_time = self.deletion_heap[0] + 1
        pod.creation_time = post_deletion_time
        new_event = (post_deletion_time, Event(EventType.CREATION, pod))
        heapq.heappush(self.event_heap, new_event)
    