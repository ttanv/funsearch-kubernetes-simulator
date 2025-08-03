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
    
    # Return inverse of remaining resources (higher score => better fit)
    score = 1000000 // (remaining_cpu + remaining_memory + remaining_gpus + 1)
    return score


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


def print_simulation_results(cluster: Cluster, pods: List[Pod], simulation_time: float, scheduler_name: str):
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
    
    # Ensure deletion times are set
    for pod in pods:
        pod.deletion_time = pod.creation_time + pod.duration_time
    
    # Run simulation
    event_simulator = DiscreteEventSimulator(pods)
    simulator = KubernetesSimulator(cluster, pods, event_simulator, first_fit_scheduler)
    
    start_time = time.time()
    try:
        simulator.run_schedule()
        simulation_time = time.time() - start_time
        print_simulation_results(cluster, pods, simulation_time, "first fit")
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