#!/usr/bin/env python3

import time
from typing import List

from benchmarks.parser import TraceParser
from simulator.entities import Cluster, Node, Pod
from simulator.event_simulator import DiscreteEventSimulator
from simulator.main import KubernetesSimulator


def best_fit_scheduler(pod: Pod, node: Node) -> int:
    """Best-fit scheduling policy: returns higher score for nodes with tighter resource fit."""
    if (pod.cpu_milli > node.cpu_milli_left or 
        pod.memory_mib > node.memory_mib_left or 
        pod.num_gpu > node.gpu_left):
        return 0
    
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
    
    # Normalize resources using typical node capacities for proper best fit scoring
    # Typical node: 64000 CPU milli, 262144 Memory MiB, 2 GPUs
    max_cpu = 64000.0
    max_memory = 262144.0
    max_gpus = 2.0
    
    # Calculate normalized remaining resources (0-1 scale)
    norm_cpu = remaining_cpu / max_cpu
    norm_memory = remaining_memory / max_memory  
    norm_gpus = remaining_gpus / max_gpus
    
    # Weighted sum with equal weights (like kubernetes-scheduler-simulator)
    weights = [0.33, 0.33, 0.34]  # cpu, memory, gpu
    normalized_remaining = (norm_cpu * weights[0] + 
                           norm_memory * weights[1] + 
                           norm_gpus * weights[2])
    
    # Return inverse of normalized remaining (higher score => better fit)
    # Scale to reasonable integer range
    score = int((1.0 - normalized_remaining) * 10000)
    return max(1, score)  # Ensure positive score


def first_fit_scheduler(pod: Pod, node: Node) -> int:
    """First-fit scheduling policy: returns fixed score if pod can fit on node."""
    if (pod.cpu_milli > node.cpu_milli_left or 
        pod.memory_mib > node.memory_mib_left or 
        pod.num_gpu > node.gpu_left):
        return 0
    
    if pod.num_gpu > 0:
        available_gpus = 0
        for gpu in node.gpus:
            if gpu.gpu_milli_left >= pod.gpu_milli:
                available_gpus += 1
        
        if available_gpus < pod.num_gpu:
            return 0
    
    # Return fixed score if pod can fit (first fit strategy)
    return 1000


def print_simulation_results(simulator: KubernetesSimulator, cluster: Cluster, pods: List[Pod], simulation_time: float, scheduler_name: str):
    """Print simulation results and metrics"""
    print(f"\n=== {scheduler_name} Results ===")
    
    # Basic scheduling metrics
    total_pods = len(pods)
    scheduled_pods = [pod for pod in pods if pod.assigned_node != ""]
    unscheduled_pods = [pod for pod in pods if pod.assigned_node == ""]
    
    success_rate = len(scheduled_pods) / total_pods * 100 if total_pods > 0 else 0
    
    print(f"Simulation completed in {simulation_time:.2f} seconds")
    print(f"Total pods: {total_pods}")
    print(f"Successfully scheduled: {len(scheduled_pods)} ({success_rate:.1f}%)")
    print(f"Failed to schedule: {len(unscheduled_pods)}")
    
    # Cluster utilization
    total_nodes = len(cluster.nodes_dict)
    nodes_with_scheduled_pods = set(pod.assigned_node for pod in scheduled_pods)
    utilized_nodes = len(nodes_with_scheduled_pods)
    
    print(f"Total nodes: {total_nodes}")
    print(f"Nodes with scheduled pods: {utilized_nodes}")
    print(f"Node utilization rate: {utilized_nodes/total_nodes*100:.1f}%")
    
    # Workload breakdown
    gpu_pods = [pod for pod in scheduled_pods if pod.num_gpu > 0]
    cpu_only_pods = [pod for pod in scheduled_pods if pod.num_gpu == 0]
    
    print(f"GPU workloads scheduled: {len(gpu_pods)}")
    print(f"CPU-only workloads scheduled: {len(cpu_only_pods)}")


def test_integration_full_dataset():
    """Test simulator on full dataset with best fit scheduling"""
    print("Testing Kubernetes Simulator on Full Dataset with Best Fit Scheduling")
    print("=" * 70)
    
    # Parse default workload from full dataset
    parser = TraceParser()
    try:
        cluster, pods = parser.parse_workload()
    except Exception as e:
        print(f"Failed to parse workload: {e}")
        return False
    
    print(f"Loaded cluster with {len(cluster.nodes_dict)} nodes")
    print(f"Loaded {len(pods)} pods")
    
    # Run simulation
    event_simulator = DiscreteEventSimulator(pods)
    simulator = KubernetesSimulator(cluster, pods, event_simulator, first_fit_scheduler, False)
    
    start_time = time.time()
    try:
        simulator.run_schedule()
        simulation_time = time.time() - start_time
        print_simulation_results(simulator, cluster, pods, simulation_time, "first fit")
        return True
    except Exception as e:
        simulation_time = time.time() - start_time
        print(f"Simulation failed after {simulation_time:.2f} seconds: {e}")
        return False


if __name__ == "__main__":
    success = test_integration_full_dataset()
    
    if success:
        print("\nIntegration test completed successfully")
    else:
        print("\nIntegration test failed")
        exit(1)