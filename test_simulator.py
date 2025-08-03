#!/usr/bin/env python3

from simulator.entities import Cluster, Node, GPU, Pod
from simulator.event_simulator import DiscreteEventSimulator
from simulator.main import KubernetesSimulator

def best_fit_scheduler(pod: Pod, node: Node) -> int:
    """Best-fit scheduling policy: returns higher score for nodes with tighter resource fit."""
    # Check if pod can fit on the node
    if (pod.cpu_milli > node.cpu_milli_left or 
        pod.memory_mib > node.memory_mib_left or 
        pod.num_gpu > node.gpu_left):
        return 0
    
    # Check GPU memory requirements
    if pod.num_gpu > 0:
        available_gpus = 0
        for gpu in node.gpus:
            if gpu.gpu_milli_left >= pod.gpu_milli:
                available_gpus += 1
        
        if available_gpus < pod.num_gpu:
            return 0
    
    # Calculate remaining resources after allocation
    remaining_cpu = node.cpu_milli_left - pod.cpu_milli
    remaining_memory = node.memory_mib_left - pod.memory_mib
    remaining_gpus = node.gpu_left - pod.num_gpu
    
    # Return inverse of remaining resources (higher score => better fit)
    score = 1000000 // (remaining_cpu + remaining_memory + remaining_gpus + 1)
    return score

def create_test_cluster():
    """Create a simple test cluster with 2 nodes"""
    # Node1 - powerful node with GPUs
    gpu1 = GPU(memory_mib_left=8000, memory_mib_total=8000, 
               gpu_milli_left=1000, gpu_milli_total=1000)
    gpu2 = GPU(memory_mib_left=8000, memory_mib_total=8000, 
               gpu_milli_left=1000, gpu_milli_total=1000)
    
    node1 = Node(node_id="node1",
                 cpu_milli_left=8000, cpu_milli_total=8000,
                 memory_mib_left=16000, memory_mib_total=16000,
                 gpu_left=2, gpus=[gpu1, gpu2])
    
    # Node2 - smaller node without GPUs
    node2 = Node(node_id="node2",
                 cpu_milli_left=4000, cpu_milli_total=4000,
                 memory_mib_left=8000, memory_mib_total=8000,
                 gpu_left=0, gpus=[])
    
    cluster = Cluster(nodes_dict={"node1": node1, "node2": node2})
    return cluster

def create_test_pods():
    """Create test pods with varying resource requirements"""
    pods = [
        Pod(pod_id="pod1", cpu_milli=1000, memory_mib=2000, 
            num_gpu=0, gpu_milli=0, gpu_spec="", 
            creation_time=0, duration_time=10,
            assigned_node="", assigned_gpus=[]),
        
        Pod(pod_id="pod2", cpu_milli=2000, memory_mib=4000, 
            num_gpu=1, gpu_milli=500, gpu_spec="v100", 
            creation_time=5, duration_time=15,
            assigned_node="", assigned_gpus=[]),
        
        Pod(pod_id="pod3", cpu_milli=3000, memory_mib=6000, 
            num_gpu=0, gpu_milli=0, gpu_spec="", 
            creation_time=10, duration_time=8,
            assigned_node="", assigned_gpus=[]),
        
        Pod(pod_id="pod4", cpu_milli=1500, memory_mib=3000, 
            num_gpu=2, gpu_milli=400, gpu_spec="v100", 
            creation_time=15, duration_time=12,
            assigned_node="", assigned_gpus=[]),
    ]
    return pods

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

def test_simulator():
    """Test the Kubernetes simulator with best-fit scheduling"""
    print("Testing Kubernetes Simulator with Best-Fit Scheduling")
    
    cluster = create_test_cluster()
    pods = create_test_pods()
    
    # Set deletion times
    for pod in pods:
        pod.deletion_time = pod.creation_time + pod.duration_time
    
    print_cluster_state(cluster, "Initial")
    
    event_simulator = DiscreteEventSimulator(pods)
    simulator = KubernetesSimulator(cluster, pods, event_simulator, best_fit_scheduler)
    
    print(f"\nStarting simulation with {len(pods)} pods...")
    
    try:
        simulator.run_schedule()
        print("Simulation completed successfully")
        
        print_cluster_state(cluster, "Final")
        
        assigned_pods = [pod for pod in pods if pod.assigned_node != ""]
        print(f"\nResults:")
        print(f"  Total pods: {len(pods)}")
        print(f"  Successfully scheduled: {len(assigned_pods)}")
        
        for pod in assigned_pods:
            print(f"  {pod.pod_id} -> {pod.assigned_node} (GPUs: {pod.assigned_gpus})")
            
    except Exception as e:
        print(f"Simulation failed: {e}")
        print_cluster_state(cluster, "Error State")
        raise

if __name__ == "__main__":
    test_simulator()