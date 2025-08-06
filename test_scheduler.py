#!/usr/bin/env python3

import time
from typing import List, Dict, Callable, Tuple
from collections import defaultdict
import statistics

from benchmarks.parser import TraceParser
from simulator.entities import Cluster, Node, Pod
from simulator.event_simulator import DiscreteEventSimulator
from simulator.main import KubernetesSimulator
from simulator.evaluator import SchedulingEvaluator

# ============= SCHEDULER IMPLEMENTATIONS =============

def gpu_first_packing_scheduler(pod: Pod, node: Node) -> int:
    """
    Optimized scheduler for GPU-heavy workloads (86.7% GPU pods).
    Strategy: Pack GPU pods tightly on GPU nodes, keep CPU pods separate.
    """
    # Basic feasibility check
    if (pod.cpu_milli > node.cpu_milli_left or 
        pod.memory_mib > node.memory_mib_left or 
        pod.num_gpu > node.gpu_left):
        return 0
    
    # GPU feasibility check
    if pod.num_gpu > 0:
        available_gpus = 0
        for gpu in node.gpus:
            if gpu.gpu_milli_left >= pod.gpu_milli:
                available_gpus += 1
        if available_gpus < pod.num_gpu:
            return 0
    
    # Calculate current node utilization
    cpu_util = 1 - (node.cpu_milli_left / node.cpu_milli_total)
    mem_util = 1 - (node.memory_mib_left / node.memory_mib_total)
    gpu_util = 1 - (node.gpu_left / max(len(node.gpus), 1)) if len(node.gpus) > 0 else 0
    overall_util = max(cpu_util, mem_util, gpu_util)
    
    node_has_gpu = len(node.gpus) > 0
    pod_needs_gpu = pod.num_gpu > 0
    
    # GPU Pod Scheduling Logic
    if pod_needs_gpu:
        if not node_has_gpu:
            return 0  # Can't place GPU pod on non-GPU node
        
        # Strong preference for nodes already running GPU workloads
        if gpu_util > 0.01:
            # Pack GPU pods together using first-fit strategy
            # Higher utilization = higher score (pack tightly)
            score = 10000 + int(gpu_util * 5000)
            
            # Bonus for good GPU utilization without exhausting CPU/memory
            if gpu_util > 0.5 and cpu_util < 0.8 and mem_util < 0.8:
                score += 3000
        else:
            # Empty GPU node - use only if necessary
            score = 5000
            
            # Penalty if CPU/memory are already heavily used (avoid mixing)
            if cpu_util > 0.3 or mem_util > 0.3:
                score -= 2000
    
    # CPU-only Pod Scheduling Logic
    else:
        if node_has_gpu:
            # Avoid GPU nodes for CPU-only workloads
            if gpu_util > 0.01:
                # GPU node with GPU workloads - strongly avoid
                return 1  # Minimal score
            else:
                # GPU node but no GPU workloads yet
                # Use only if it already has CPU workloads
                if cpu_util > 0.1:
                    score = 3000
                else:
                    score = 500  # Low priority
        else:
            # Non-GPU node - perfect for CPU-only workloads
            # Use first-fit strategy within non-GPU nodes
            score = 15000
            
            # Pack CPU pods together
            if cpu_util > 0.1:
                score += int(cpu_util * 3000)
    
    return max(1, score)


def temporal_aware_scheduler(pod: Pod, node: Node) -> int:
    """
    Scheduler that considers pod duration for better temporal packing.
    Tries to co-locate pods with similar lifetimes.
    """
    # Basic feasibility check
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
    
    # Calculate utilization
    cpu_util = 1 - (node.cpu_milli_left / node.cpu_milli_total)
    mem_util = 1 - (node.memory_mib_left / node.memory_mib_total)
    gpu_util = 1 - (node.gpu_left / max(len(node.gpus), 1)) if len(node.gpus) > 0 else 0
    
    # Categorize pod by duration (assuming we can access duration_time)
    # Short: < 1000 time units, Medium: 1000-5000, Long: > 5000
    duration_category = "short" if pod.duration_time < 1000 else ("medium" if pod.duration_time < 5000 else "long")
    
    base_score = 1000
    
    # Strategy: Pack pods with similar durations together
    if duration_category == "short":
        # Short pods: aggressive packing, prefer partially filled nodes
        if 0.3 < cpu_util < 0.9:
            base_score += 8000
        else:
            base_score += 2000
    elif duration_category == "medium":
        # Medium pods: balanced approach
        if 0.2 < cpu_util < 0.7:
            base_score += 6000
        else:
            base_score += 3000
    else:  # long
        # Long pods: prefer empty or nearly empty nodes
        if cpu_util < 0.2:
            base_score += 7000
        else:
            base_score += 1000
    
    # GPU-specific adjustments
    if pod.num_gpu > 0 and len(node.gpus) > 0:
        if gpu_util > 0.1:
            base_score += 3000  # Pack GPU pods together
    elif pod.num_gpu == 0 and len(node.gpus) == 0:
        base_score += 2000  # CPU pods on CPU nodes
    
    return max(1, base_score)


def compact_first_fit_scheduler(pod: Pod, node: Node) -> int:
    """
    Enhanced first-fit that maintains node ordering but with slight preference
    for partially filled nodes to reduce fragmentation.
    """
    # Basic feasibility check
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
    
    # Extract node number from node_id (e.g., "openb-node-0000" -> 0)
    try:
        node_number = int(node.node_id.split('-')[-1])
    except:
        node_number = hash(node.node_id) % 10000
    
    # Calculate utilization
    cpu_util = 1 - (node.cpu_milli_left / node.cpu_milli_total)
    mem_util = 1 - (node.memory_mib_left / node.memory_mib_total)
    overall_util = max(cpu_util, mem_util)
    
    # First-fit base score inversely proportional to node number
    # This ensures nodes are filled in order
    base_score = 10000 - node_number
    
    # Small bonus for partially filled nodes (compaction)
    if 0.1 < overall_util < 0.9:
        base_score += 500
    
    # Extra bonus for nearly full nodes (reduce fragmentation)
    if overall_util > 0.7:
        base_score += 300
    
    return max(1, base_score)

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


def worst_fit_scheduler(pod: Pod, node: Node) -> int:
    """Worst-fit: prefers nodes with most remaining resources."""
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
    
    remaining_cpu = (node.cpu_milli_left - pod.cpu_milli) / node.cpu_milli_total
    remaining_memory = (node.memory_mib_left - pod.memory_mib) / node.memory_mib_total
    remaining_gpus = (node.gpu_left - pod.num_gpu) / max(len(node.gpus), 1)
    
    weights = [0.33, 0.33, 0.34]
    score = int((remaining_cpu * weights[0] + 
                 remaining_memory * weights[1] + 
                 remaining_gpus * weights[2]) * 10000)
    return max(1, score)


def hybrid_scheduler(pod: Pod, node: Node) -> int:
    """Hybrid scheduler: worst-fit for large pods, best-fit for small pods."""
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
    
    # Calculate pod size
    pod_cpu_ratio = pod.cpu_milli / node.cpu_milli_total
    pod_memory_ratio = pod.memory_mib / node.memory_mib_total
    pod_gpu_ratio = pod.num_gpu / max(len(node.gpus), 1)
    pod_size = max(pod_cpu_ratio, pod_memory_ratio, pod_gpu_ratio)
    
    # Calculate remaining resources
    remaining_cpu = (node.cpu_milli_left - pod.cpu_milli) / node.cpu_milli_total
    remaining_memory = (node.memory_mib_left - pod.memory_mib) / node.memory_mib_total
    remaining_gpus = (node.gpu_left - pod.num_gpu) / max(len(node.gpus), 1)
    
    # Current utilization
    current_util = 1 - min(node.cpu_milli_left / node.cpu_milli_total,
                           node.memory_mib_left / node.memory_mib_total)
    
    if pod_size > 0.3:  # Large pods - use worst-fit
        score = int((remaining_cpu + remaining_memory + remaining_gpus) * 3333)
        if current_util < 0.01:  # Bonus for empty nodes
            score += 5000
    else:  # Small pods - use best-fit for gap filling
        if 0.3 < current_util < 0.9:
            score = int((1 - (remaining_cpu + remaining_memory + remaining_gpus) / 3) * 10000)
            score += 2000
        elif current_util >= 0.9:
            score = 100 if pod_size >= 0.1 else 8000
        else:
            score = 100
    
    return max(1, score)


def workload_aware_scheduler(pod: Pod, node: Node) -> int:
    """Segregates GPU and CPU workloads for better packing."""
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
    
    node_has_gpu = len(node.gpus) > 0
    pod_needs_gpu = pod.num_gpu > 0
    
    cpu_util = 1 - (node.cpu_milli_left / node.cpu_milli_total)
    mem_util = 1 - (node.memory_mib_left / node.memory_mib_total)
    gpu_util = 1 - (node.gpu_left / max(len(node.gpus), 1)) if node_has_gpu else 0
    
    remaining_cpu_norm = (node.cpu_milli_left - pod.cpu_milli) / node.cpu_milli_total
    remaining_mem_norm = (node.memory_mib_left - pod.memory_mib) / node.memory_mib_total
    
    base_score = 1000
    
    if pod_needs_gpu:
        if not node_has_gpu:
            return 0
        if gpu_util > 0.1:
            base_score += 8000
            remaining_gpu_norm = (node.gpu_left - pod.num_gpu) / len(node.gpus)
            score = base_score + int((1 - remaining_gpu_norm) * 5000)
        else:
            score = base_score + 2000
        if cpu_util > 0.1 and gpu_util < 0.1:
            score = max(1, score - 5000)
    else:
        if node_has_gpu:
            score = 100 if gpu_util > 0.1 else base_score
        else:
            base_score += 5000
            if cpu_util > 0.2:
                score = base_score + int((1 - (remaining_cpu_norm + remaining_mem_norm) / 2) * 4000)
            else:
                score = base_score + 2000
    
    cpu_mem_balance = 1 - abs(remaining_cpu_norm - remaining_mem_norm)
    score += int(cpu_mem_balance * 1000)
    
    return max(1, score)


# ============= TESTING FRAMEWORK =============

class SchedulerTester:
    def __init__(self, cluster: Cluster, pods: List[Pod]):
        self.original_cluster = cluster
        self.original_pods = pods
        self.schedulers = {
            'first_fit': first_fit_scheduler,
            'best_fit': best_fit_scheduler,
            'worst_fit': worst_fit_scheduler,
            'hybrid': hybrid_scheduler,
            'workload_aware': workload_aware_scheduler,
            'gpu_first': gpu_first_packing_scheduler,
            'temporal_aware': temporal_aware_scheduler,
            'compact_first': compact_first_fit_scheduler,
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
        }
        
        # Run simulation
        start_time = time.time()
        try:
            simulator.run_schedule()
            metrics['simulation_time'] = time.time() - start_time
            
            # Get evaluation results
            metrics['evaluation_results'] = simulator.get_evaluation_results()
            
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
            
            if eval_results:
                print(f"  Average CPU Utilization:  {eval_results.avg_cpu_utilization:.1%}")
                print(f"  Average Memory Utilization: {eval_results.avg_memory_utilization:.1%}")
                print(f"  Average GPU Count Util:   {eval_results.avg_gpu_count_utilization:.1%}")
                print(f"  Average GPU Memory Util:  {eval_results.avg_gpu_memory_utilization:.1%}")
                print(f"  GPU Fragmentation Score:  {eval_results.gpu_fragmentation_score:.3f}")
                print(f"  Utilization Snapshots:    {eval_results.num_snapshots}")
                print(f"  Fragmentation Events:     {eval_results.num_fragmentation_events}")
            else:
                print("  Evaluation Results:       Not available")
            
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