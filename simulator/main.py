from typing import Callable, Optional

from simulator.entities import Cluster, Node, GPU, Pod
from simulator.event_simulator import DiscreteEventSimulator, Event, EventType
from simulator.evaluator import SchedulingEvaluator

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
                 scheduler: PodNodeScorer,
                 validate_invariants: bool = True,
                 evaluator: Optional[SchedulingEvaluator] = None):
        self.cluster = cluster
        self.pod_list = pod_list
        self.event_simulator = event_simulator
        self.scheduler = scheduler
        self.validate_invariants = validate_invariants
        self.evaluator = evaluator
        self.max_nodes = 0
        self.waiting_pods = []  # Track pods that failed to schedule
        
        # Initialize evaluator with total events count
        if self.evaluator:
            total_events = len(event_simulator.event_heap)
            self.evaluator.initialize(total_events)
             
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
                
            # Record event processed for evaluation
            if self.evaluator:
                self.evaluator.record_event_processed(self.cluster)
                
            # Update max_nodes with current number of active nodes
            active_nodes = len([node for node in self.cluster.nodes_dict.values() 
                              if (node.cpu_milli_left < node.cpu_milli_total or 
                                  node.memory_mib_left < node.memory_mib_total or 
                                  node.gpu_left < len(node.gpus))])
            self.max_nodes = max(self.max_nodes, active_nodes)
            
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
            if self.validate_invariants:
                self._validate_cluster_invariants()
            return
        
        for gpu_id in gpu_ids:
            gpu: GPU = node.gpus[gpu_id]
            gpu.gpu_milli_left += pod.gpu_milli
        
        if self.validate_invariants:
            self._validate_cluster_invariants() 
            
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
            # Add to waiting pods and record fragmentation event for evaluation
            if pod not in self.waiting_pods:
                self.waiting_pods.append(pod)
            
            if self.evaluator:
                self.evaluator.record_fragmentation_event(self.cluster, self.waiting_pods)
            
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
        
        # Remove from waiting pods if it was there
        if pod in self.waiting_pods:
            self.waiting_pods.remove(pod)
        
        # Create deletion event
        self.event_simulator.push_deletion_event(pod)
        
        if self.validate_invariants:
            self._validate_cluster_invariants()
    
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
    
    def _validate_cluster_invariants(self):
        """Validate cluster state invariants to ensure correctness for debudding purposes"""
        for node_id, node in self.cluster.nodes_dict.items():
            # Check node resource invariants
            if node.cpu_milli_left < 0:
                raise ValueError(f"Node {node_id} has negative CPU remaining: {node.cpu_milli_left}")
            if node.memory_mib_left < 0:
                raise ValueError(f"Node {node_id} has negative memory remaining: {node.memory_mib_left}")
            if node.gpu_left < 0:
                raise ValueError(f"Node {node_id} has negative GPU count remaining: {node.gpu_left}")
            
            # Check that remaining resources don't exceed total
            if node.cpu_milli_left > node.cpu_milli_total:
                raise ValueError(f"Node {node_id} CPU remaining exceeds total: {node.cpu_milli_left} > {node.cpu_milli_total}")
            if node.memory_mib_left > node.memory_mib_total:
                raise ValueError(f"Node {node_id} memory remaining exceeds total: {node.memory_mib_left} > {node.memory_mib_total}")
            if node.gpu_left > len(node.gpus):
                raise ValueError(f"Node {node_id} GPU remaining exceeds total: {node.gpu_left} > {len(node.gpus)}")
            
            # Check GPU invariants
            for i, gpu in enumerate(node.gpus):
                if gpu.gpu_milli_left < 0:
                    raise ValueError(f"Node {node_id} GPU {i} has negative milli remaining: {gpu.gpu_milli_left}")
                if gpu.gpu_milli_left > gpu.gpu_milli_total:
                    raise ValueError(f"Node {node_id} GPU {i} milli remaining exceeds total: {gpu.gpu_milli_left} > {gpu.gpu_milli_total}")
        
        # Check pods assigned to nodes have valid node assignments
        for pod in self.pod_list:
            if pod.assigned_node != "" and pod.assigned_node not in self.cluster.nodes_dict:
                raise ValueError(f"Pod {pod.pod_id} assigned to non-existent node: {pod.assigned_node}")
        
        # Consider only valid pods
        valid_pods = []
        for _, event in self.event_simulator.event_heap:
            if event.pod.assigned_node != "":
                valid_pods.append(event.pod)     
                
        # Validate resource accounting: used + remaining = total for each node
        for node_id, node in self.cluster.nodes_dict.items():
            # Find all pods assigned to this node
            assigned_pods = [pod for pod in valid_pods if pod.assigned_node == node_id]
            
            # Calculate total resources used by assigned pods
            used_cpu = sum(pod.cpu_milli for pod in assigned_pods)
            used_memory = sum(pod.memory_mib for pod in assigned_pods)
            used_gpus = sum(pod.num_gpu for pod in assigned_pods)
            
            # Check that used + remaining = total
            if used_cpu + node.cpu_milli_left != node.cpu_milli_total:
                raise ValueError(f"Node {node_id} CPU accounting error: used({used_cpu}) + "
                                f"remaining({node.cpu_milli_left}) != total({node.cpu_milli_total})")
            if used_memory + node.memory_mib_left != node.memory_mib_total:
                raise ValueError(f"Node {node_id} memory accounting error: used({used_memory}) + "
                                f"remaining({node.memory_mib_left}) != total({node.memory_mib_total})")
            if used_gpus + node.gpu_left != len(node.gpus):
                raise ValueError(f"Node {node_id} GPU accounting error: used({used_gpus}) + "
                                f"remaining({node.gpu_left}) != total({len(node.gpus)})")
            
            # Check GPU milli accounting
            gpu_milli_used = {}
            for pod in assigned_pods:
                if pod.num_gpu > 0:
                    for gpu_index in pod.assigned_gpus:
                        if gpu_index not in gpu_milli_used:
                            gpu_milli_used[gpu_index] = 0
                        gpu_milli_used[gpu_index] += pod.gpu_milli
            
            for i, gpu in enumerate(node.gpus):
                used_gpu_milli = gpu_milli_used.get(i, 0)
                if used_gpu_milli + gpu.gpu_milli_left != gpu.gpu_milli_total:
                    raise ValueError(f"Node {node_id} GPU {i} milli accounting error: used({used_gpu_milli}) + "
                                    "remaining({gpu.gpu_milli_left}) != total({gpu.gpu_milli_total})")
    
    def get_evaluation_results(self):
        """Get evaluation results from the evaluator if enabled"""
        if self.evaluator:
            return self.evaluator.get_evaluation_results()
        return None