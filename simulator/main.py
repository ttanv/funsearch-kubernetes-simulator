from typing import Callable

from simulator.entities import Cluster, Node, GPU, Pod
from simulator.event_simulator import DiscreteEventSimulator, Event, EventType

# Type alias for a function that takes a Pod and a Node and returns an int
PodNodeScorer = Callable[[Pod, Node], int]

def print_cluster_state(cluster: Cluster, step: str):
    """Print current cluster resource usage"""
    print(f"\n--- Cluster State: {step} ---")
    for node_id, node in cluster.nodes_dict.items():
        cpu_used = node.cpu_milli_total - node.cpu_milli_left
        memory_used = node.memory_mib_total - node.memory_mib_left
        gpu_used = len(node.gpus) - node.gpu_left
        
        print(f"{node_id}:")
        print(f"  CPU: {cpu_used}/{node.cpu_milli_total} milli")
        print(f"  Memory: {memory_used}/{node.memory_mib_total} MiB")
        print(f"  GPUs: {gpu_used}/{len(node.gpus)}")
        
        for i, gpu in enumerate(node.gpus):
            gpu_mem_used = gpu.gpu_milli_total - gpu.gpu_milli_left
            print(f"    GPU{i}: {gpu_mem_used}/{gpu.gpu_milli_total} milli")

class KubernetesSimulator:
    def __init__(self, 
                 cluster: Cluster, 
                 pod_list: list[Pod],
                 event_simulator: DiscreteEventSimulator,
                 scheduler: PodNodeScorer):
        self.cluster = cluster
        self.pod_list = pod_list
        self.event_simulator = event_simulator
        self.scheduler = scheduler
             
    def run_schedule(self):
        """
        Runs the scheduler end to end
        """
        while not self.event_simulator.finished_events():
            # Get the next event
            time, event = self.event_simulator.pop_event()
            
            if event.event_type == EventType.DELETION:
                self._handle_deletion(event)
            elif event.event_type == EventType.CREATION:
                self._handle_creation(event)
            
    def _handle_deletion(self, event: Event):
        pod: Pod = event.pod
        node_id = pod.assigned_node
        
        if node_id == "":
            raise ValueError("Invalid node id, pod was never assigned node yet being deleted")
        
        # Free up node resources
        node = self.cluster.nodes_dict[node_id]
        node.cpu_milli_left += pod.cpu_milli
        node.memory_mib_left += pod.memory_mib
        node.gpu_left += pod.num_gpu
        
        # Free up gpu resources if valid
        gpu_ids = pod.assigned_gpus
        if len(gpu_ids) == 0:
            return
        
        for gpu_id in gpu_ids:
            gpu: GPU = node.gpus[gpu_id]
            gpu.gpu_milli_left += pod.gpu_milli 
            
    def _handle_creation(self, event: Event):
        pod: Pod = event.pod
        
        best_score = 0
        best_node: Node = None
        for node in self.cluster.nodes_dict.values():
            curr_score = self.scheduler(pod, node)
            
            if curr_score > best_score:
                best_score = curr_score
                best_node = node
        
        # If not suitable such node, reschedule until after next deletion
        if best_node is None:
            self.event_simulator.repush_creation_event(pod)
            return
        
        # Update the node
        best_node.cpu_milli_left -= pod.cpu_milli
        best_node.memory_mib_left -= pod.memory_mib
        best_node.gpu_left -= pod.num_gpu
        
        # Choose gpus from node, and update those gpus' info
        allocated_gpu_indices = self._allocate_gpus_best_fit(best_node, pod)
        
        # Alternative below, may make the simulation faster?
        # allocated_gpu_indices = self._allocate_gpus_first_fit(best_node, pod)
        
        # Update the pod info
        pod.assigned_node = best_node.node_id
        pod.assigned_gpus = allocated_gpu_indices
        
        # Create deletion event
        self.event_simulator.push_deletion_event(pod)
        
        # print_cluster_state(self.cluster, "During Run")
    
    def _allocate_gpus_best_fit(self, node: Node, pod: Pod) -> list[int]:
        """
        Best fit GPU allocation: selects GPUs with the least available memory 
        that can still satisfy the pod's requirements.
        """
        if pod.num_gpu == 0:
            return []
        
        # Find GPUs that can satisfy the pod's GPU memory requirement
        available_gpus = []
        for i, gpu in enumerate(node.gpus):
            if gpu.gpu_milli_left >= pod.gpu_milli:
                available_gpus.append((i, gpu.gpu_milli_left))
        
        if len(available_gpus) < pod.num_gpu:
            raise ValueError(f"Not enough GPUs available on node {node.node_id}")
        
        # Sort by available memory (ascending) for best fit
        available_gpus.sort(key=lambda x: x[1])
        
        # Allocate the required number of GPUs
        allocated_indices = []
        for i in range(pod.num_gpu):
            gpu_index = available_gpus[i][0]
            allocated_indices.append(gpu_index)
            node.gpus[gpu_index].gpu_milli_left -= pod.gpu_milli
        
        return allocated_indices
    
    def _allocate_gpus_first_fit(self, node: Node, pod: Pod) -> list[int]:
        """
        First fit GPU allocation: selects the first available GPUs.
        """
        if pod.num_gpu == 0:
            return []
        
        allocated_indices = []
        for i, gpu in enumerate(node.gpus):
            if gpu.gpu_milli_left >= pod.gpu_milli:
                allocated_indices.append(i)
                # Update GPU memory
                gpu.gpu_milli_left -= pod.gpu_milli
                
                if len(allocated_indices) == pod.num_gpu:
                    break
        
        if len(allocated_indices) < pod.num_gpu:
            raise ValueError(f"Not enough GPUs available on node {node.node_id}")
        
        return allocated_indices