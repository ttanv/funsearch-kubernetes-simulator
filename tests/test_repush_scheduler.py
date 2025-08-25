#!/usr/bin/env python3

import sys
import os
import time
from typing import List, Dict, Callable


# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.parser import TraceParser
from simulator.entities import Cluster, Node, Pod
from simulator.event_simulator import DiscreteEventSimulator
from simulator.main import KubernetesSimulator
from simulator.repush_evaluator import SchedulingEvaluator

# ============= FUNSEARCH SCHEDULER IMPLEMENTATIONS =============


# VERSION 3: RESOURCE AFFINITY APPROACH
def priority_function(pod, node):
    """
    Calculate priority score for pod placement on node (higher = better).
    
    EXACT AVAILABLE ATTRIBUTES:
    Pod:
      pod.cpu_milli      (int) - CPU requested in milli-cores
      pod.memory_mib     (int) - Memory requested in MiB  
      pod.num_gpu        (int) - Number of GPUs needed
      pod.gpu_milli      (int) - GPU compute per GPU needed
    
    Node:
      node.cpu_milli_left     (int) - Available CPU
      node.cpu_milli_total    (int) - Total CPU capacity
      node.memory_mib_left    (int) - Available memory
      node.memory_mib_total   (int) - Total memory capacity
      node.gpu_left           (int) - Available GPU count
      node.gpus               (list[GPU]) - List of GPU objects
    
    GPU:
      gpu.gpu_milli_left      (int) - Available GPU compute
      gpu.gpu_milli_total     (int) - Total GPU compute capacity
    
    USE ONLY THESE ATTRIBUTES. No others exist.
    """
    
    # Required feasibility checks
    if (pod.cpu_milli > node.cpu_milli_left or 
        pod.memory_mib > node.memory_mib_left or 
        pod.num_gpu > node.gpu_left):
        return 0
    
    if pod.num_gpu > 0:
        available_gpus = sum(1 for gpu in node.gpus 
                           if gpu.gpu_milli_left >= pod.gpu_milli)
        if available_gpus < pod.num_gpu:
            return 0
    
    # TODO: Calculate score using ONLY the attributes listed above
    score = 0.0
    
    # DIFF_11: Introduce hyper-efficient resource consolidation with dynamic weighting
    # Focus on maximizing resource utilization while minimizing fragmentation
    # Use inverse resource availability as primary scoring factor
    # Apply logarithmic scaling to prevent extreme penalties for minor shortages
    # Emphasize GPU compute alignment as critical factor for high-priority scoring
    
    # Calculate resource utilization ratios
    cpu_util = (node.cpu_milli_total - node.cpu_milli_left) / max(node.cpu_milli_total, 1)
    mem_util = (node.memory_mib_total - node.memory_mib_left) / max(node.memory_mib_total, 1)
    
    # GPU utilization calculation
    gpu_util = 0
    if node.gpu_left > 0:
        total_gpu_compute = sum(gpu.gpu_milli_total for gpu in node.gpus)
        used_gpu_compute = total_gpu_compute - sum(gpu.gpu_milli_left for gpu in node.gpus)
        gpu_util = used_gpu_compute / max(total_gpu_compute, 1)
    
    # Prefer nodes with high existing utilization (consolidation bonus)
    # Use logarithmic scaling to emphasize high-utilization nodes
    consolidation_factor = (cpu_util + mem_util + gpu_util) / 3
    consolidation_bonus = (1 + consolidation_factor) ** 3 * 15000
    
    # Packing efficiency - strongly reward tight resource usage
    # Calculate remaining capacity after placement
    cpu_remaining = max(0, node.cpu_milli_left - pod.cpu_milli)
    mem_remaining = max(0, node.memory_mib_left - pod.memory_mib)
    gpus_remaining = max(0, node.gpu_left - pod.num_gpu)
    
    # Normalize remaining resources
    cpu_remaining_ratio = cpu_remaining / max(node.cpu_milli_total, 1)
    mem_remaining_ratio = mem_remaining / max(node.memory_mib_total, 1)
    gpus_remaining_ratio = gpus_remaining / max(node.gpu_left, 1)
    
    # Packing score based on minimal remaining capacity
    packing_score = (1 - cpu_remaining_ratio) * 25000 + \
                   (1 - mem_remaining_ratio) * 25000 + \
                   (1 - gpus_remaining_ratio) * 20000
    
    # GPU compute alignment bonus - critical for GPU workloads
    gpu_alignment_bonus = 0
    if pod.num_gpu > 0:
        # Check if we can satisfy GPU compute requirement
        total_available_compute = sum(gpu.gpu_milli_left for gpu in node.gpus)
        needed_compute = pod.num_gpu * pod.gpu_milli
        if total_available_compute >= needed_compute:
            # Reward precise compute matching
            match_ratio = min(1.0, needed_compute / max(total_available_compute, 1))
            gpu_alignment_bonus = (match_ratio ** 2) * 35000
    
    # Resource balance penalty - discourage heavily skewed resource usage
    util_ratios = [cpu_util, mem_util, gpu_util]
    if len(util_ratios) > 1:
        avg_util = sum(util_ratios) / len(util_ratios)
        variance = sum((r - avg_util)**2 for r in util_ratios) / len(util_ratios)
        balance_penalty = variance * 10000
    else:
        balance_penalty = 0
    
    # Fragmentation penalty - discourage uneven remaining resource distribution
    remaining_ratios = [cpu_remaining_ratio, mem_remaining_ratio, gpus_remaining_ratio]
    if len(remaining_ratios) > 1:
        avg_remaining = sum(remaining_ratios) / len(remaining_ratios)
        frag_variance = sum((r - avg_remaining)**2 for r in remaining_ratios) / len(remaining_ratios)
        fragmentation_penalty = frag_variance * 15000
    else:
        fragmentation_penalty = 0
    
    # Base score calculation
    score = consolidation_bonus + packing_score + gpu_alignment_bonus - balance_penalty - fragmentation_penalty
    
    # Tie-breaking based on resource requirements
    tie_breaker = (pod.cpu_milli + pod.memory_mib + pod.num_gpu) % 25
    score += tie_breaker * 1200
    
    # Small bonus for nodes with moderate remaining capacity
    headroom_bonus = min(cpu_remaining_ratio + mem_remaining_ratio + gpus_remaining_ratio, 0.5) * 3000
    score += headroom_bonus
    
    return max(1, int(score))

# ============= DEFAULT SCHEDULER IMPLEMENTATIONS =============

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
    
    remaining_cpu = node.cpu_milli_left - pod.cpu_milli
    remaining_memory = node.memory_mib_left - pod.memory_mib
    remaining_gpus = node.gpu_left - pod.num_gpu
    
    norm_cpu = remaining_cpu / node.cpu_milli_total
    norm_memory = remaining_memory / node.memory_mib_total
    norm_gpus = remaining_gpus / max(len(node.gpus), 1)
    
    weights = [0.33, 0.33, 0.34]
    normalized_remaining = (norm_cpu * weights[0] + 
                           norm_memory * weights[1] + 
                           norm_gpus * weights[2])
    
    score = int((1 - normalized_remaining) * 10000)
    return max(1, score)


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
    
    return 1000


# ============= TESTING FRAMEWORK =============

class SchedulerTester:
    def __init__(self, cluster: Cluster, pods: List[Pod]):
        self.original_cluster = cluster
        self.original_pods = pods
        self.schedulers = {
            'first_fit': first_fit_scheduler,
            'best_fit': best_fit_scheduler,
            'funsearch_func': priority_function,
        }
        
    def deep_copy_cluster(self, cluster: Cluster) -> Cluster:
        """Create a deep copy of the cluster"""
        import copy
        return copy.deepcopy(cluster)
    
    def deep_copy_pods(self, pods: List[Pod]) -> List[Pod]:
        """Create a deep copy of pods list"""
        import copy
        return copy.deepcopy(pods)
    
    def run_single_test(self, scheduler_name: str, scheduler_func: Callable) -> Dict:
        """Run a single test with detailed metrics"""
        # Create fresh copies for this test
        cluster = self.deep_copy_cluster(self.original_cluster)
        pods = self.deep_copy_pods(self.original_pods)
        
        # Initialize simulator with evaluator
        event_simulator = DiscreteEventSimulator(pods)
        evaluator = SchedulingEvaluator(cluster, enabled=True)
        simulator = KubernetesSimulator(cluster, pods, event_simulator, scheduler_func, evaluator=evaluator)
        
        # Track core metrics
        metrics = {
            'name': scheduler_name,
            'scheduled_pods': 0,
            'simulation_time': 0,
            'evaluation_results': None,
            'policy_score': 0.0,
        }
        
        # Run simulation
        start_time = time.time()
        try:
            simulator.run_schedule()
            metrics['simulation_time'] = time.time() - start_time
            
            # Get evaluation results
            metrics['evaluation_results'] = simulator.get_evaluation_results()
            
            # Get policy score
            metrics['policy_score'] = simulator.evaluator.get_policy_score(pods)
            
            # Count scheduled pods
            for pod in pods:
                if pod.assigned_node != "":
                    metrics['scheduled_pods'] += 1
            
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics
    
    def compare_all_schedulers(self) -> None:
        """Run all schedulers and compare results"""
        results = []
        
        print("=" * 80)
        print("SCHEDULER EVALUATION WITH NEW METRICS")
        print("=" * 80)
        print(f"Testing {len(self.schedulers)} schedulers with {len(self.original_pods)} pods on {len(self.original_cluster.nodes_dict)} nodes")
        print()
        
        # Run each scheduler
        for name, func in self.schedulers.items():
            print(f"Testing {name}...", end=" ")
            metrics = self.run_single_test(name, func)
            results.append(metrics)
            print("Done!")
        
        # Print detailed results with new metrics
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        for metrics in results:
            success_rate = metrics['scheduled_pods'] / len(self.original_pods) * 100
            eval_results = metrics['evaluation_results']
            
            print(f"\n{metrics['name'].upper()}")
            print("-" * 50)
            print(f"  Scheduled Pods:           {metrics['scheduled_pods']:4d}/{len(self.original_pods)} ({success_rate:5.1f}%)")
            print(f"  Simulation Time:          {metrics['simulation_time']:.2f}s")
            print(f"  Policy Score (0-1):       {metrics['policy_score']:.4f}")
            
            if eval_results:
                print(f"  GPU Fragmentation Score:    {eval_results.gpu_fragmentation_score:.3f}")
                print(f"  Peak Nodes Used:            {eval_results.peak_nodes_used}")
                print(f"  Total Scheduling Attempts:  {eval_results.total_scheduling_attempts}")
                print(f"  Total Repushes:             {eval_results.total_repush_events}")
                print(f"  Fragmentation Events:       {eval_results.num_fragmentation_events}")
            else:
                print("  Evaluation Results:         Not available")

            if 'error' in metrics:
                print(f"  ERROR: {metrics['error']}")
        
        
        return results


    

def main():
    """Main test runner"""
    print("Loading workload data...")
    parser = TraceParser()
    
    try:
        cluster, pods = parser.parse_workload()
    except Exception as e:
        print(f"Failed to parse workload: {e}")
        return
    
    print(f"Loaded {len(cluster.nodes_dict)} nodes and {len(pods)} pods")
    
    # Run tests
    tester = SchedulerTester(cluster, pods)
    results = tester.compare_all_schedulers()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()