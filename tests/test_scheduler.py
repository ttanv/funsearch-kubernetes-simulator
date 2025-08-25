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
from simulator.evaluator import SchedulingEvaluator

# ============= FUNSEARCH SCHEDULER IMPLEMENTATIONS =============

def funsearch_v1_scheduler(pod: Pod, node: Node) -> int:
    """FunSearch discovered policy with score 0.4901"""
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
    
    score = 0.0
    
    # Calculate CPU utilization score
    cpu_utilization = (node.cpu_milli_total - node.cpu_milli_left) / node.cpu_milli_total
    if cpu_utilization < 0.7:
        cpu_score = (1.0 - cpu_utilization) * 100
    else:
        cpu_score = (1.0 - cpu_utilization) * 50

    # Calculate memory utilization score
    memory_utilization = (node.memory_mib_total - node.memory_mib_left) / node.memory_mib_total
    if memory_utilization < 0.7:
        memory_score = (1.0 - memory_utilization) * 100
    else:
        memory_score = (1.0 - memory_utilization) * 50

    # Calculate GPU utilization score
    if pod.num_gpu > 0:
        gpu_utilization = (node.gpu_left * node.gpus[0].gpu_milli_total - sum(gpu.gpu_milli_left for gpu in node.gpus)) / (node.gpu_left * node.gpus[0].gpu_milli_total)
        if gpu_utilization < 0.7:
            gpu_score = (1.0 - gpu_utilization) * 200
        else:
            gpu_score = (1.0 - gpu_utilization) * 100
    else:
        gpu_score = 0

    # Combine scores
    score = cpu_score + memory_score + gpu_score

    # Penalize fragmentation for GPU nodes
    if pod.num_gpu > 0:
        free_gpu_millis = sum(gpu.gpu_milli_left for gpu in node.gpus)
        gpu_fragmentation_factor = free_gpu_millis % pod.gpu_milli
        score -= gpu_fragmentation_factor * 0.2

    # Penalize low CPU and memory capacity
    if node.cpu_milli_total < 2000 or node.memory_mib_total < 12:
        score -= (2000 - node.cpu_milli_total) * 0.01
        score -= (12 - node.memory_mib_total) * 0.1

    # Encourage balanced node loading
    balance_factor = abs(node.cpu_milli_left / max(1, node.memory_mib_left) - pod.cpu_milli / max(1, pod.memory_mib))
    score -= balance_factor * 0.5

    # Bonus for nodes with ample resources
    if node.cpu_milli_left > pod.cpu_milli * 2 and node.memory_mib_left > pod.memory_mib * 2:
        score += 25

    # Penalize nodes with unequal GPU resource left
    if pod.num_gpu > 0:
        gpu_imbalance = max(gpu.gpu_milli_left for gpu in node.gpus) - min(gpu.gpu_milli_left for gpu in node.gpus)
        score -= gpu_imbalance * 0.05

    # Bonus for nodes with high resources
    if node.cpu_milli_total > 10000 and node.memory_mib_total > 64:
        score += 15

    # Penalize if node is nearly full
    if cpu_utilization > 0.9 or memory_utilization > 0.9:
        score -= 20
    
    return max(1, int(score))


def funsearch_v2_scheduler(pod: Pod, node: Node) -> int:
    """FunSearch discovered policy with score 0.4816 - balance and efficiency focused."""
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
    
    score = 0.0
    
    cpu_util = (node.cpu_milli_total - node.cpu_milli_left + pod.cpu_milli) / max(1, node.cpu_milli_total)
    mem_util = (node.memory_mib_total - node.memory_mib_left + pod.memory_mib) / max(1, node.memory_mib_total)
    balance_score = 1 - abs(cpu_util - mem_util)
    efficiency_score = (cpu_util * mem_util) ** 0.5
    
    if pod.num_gpu > 0:
        eligible_gpus = [g for g in node.gpus if g.gpu_milli_left >= pod.gpu_milli][:pod.num_gpu]
        gpu_util = sum((g.gpu_milli_total - g.gpu_milli_left + pod.gpu_milli) for g in eligible_gpus) / max(1, sum(g.gpu_milli_total for g in eligible_gpus))
        gpu_frag = sum((g.gpu_milli_left - pod.gpu_milli) ** 2 for g in eligible_gpus) / max(1, sum(g.gpu_milli_left for g in eligible_gpus))
        isolation_score = 0.5 - abs(0.5 - (gpu_frag ** 0.5))
        score = (cpu_util * 0.25 + mem_util * 0.15 + gpu_util * 0.45 + balance_score * 0.05 + efficiency_score * 0.05 - gpu_frag * 0.05 + isolation_score * 0.1) * 10000
    else:
        frag_score = min((node.cpu_milli_left % max(1, pod.cpu_milli)) / node.cpu_milli_total, (node.memory_mib_left % max(1, pod.memory_mib)) / node.memory_mib_total)
        score = (cpu_util * 0.45 + mem_util * 0.35 + balance_score * 0.1 + efficiency_score * 0.1 - frag_score * 0.1) * 10000
    
    return max(1, int(score))


def funsearch_v3_scheduler(pod: Pod, node: Node) -> int:
    """FunSearch discovered policy with score 0.4800 - GPU optimization with balance."""
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
    
    score = 0.0
    
    cpu_util = (node.cpu_milli_total - node.cpu_milli_left + pod.cpu_milli) / node.cpu_milli_total
    mem_util = (node.memory_mib_total - node.memory_mib_left + pod.memory_mib) / node.memory_mib_total
    balance_score = (1 - abs(cpu_util - mem_util)) ** 2.5 * 300
    
    gpu_score = 0
    if pod.num_gpu > 0:
        viable_gpus = sorted([g for g in node.gpus if g.gpu_milli_left >= pod.gpu_milli], key=lambda g: g.gpu_milli_left)
        if len(viable_gpus) >= pod.num_gpu:
            gpu_eff = sum(1 - (g.gpu_milli_left - pod.gpu_milli) / g.gpu_milli_total for g in viable_gpus[:pod.num_gpu]) / pod.num_gpu
            gpu_score = (gpu_eff ** 2) * 450
    
    frag_score = min(node.cpu_milli_left - pod.cpu_milli, node.memory_mib_left - pod.memory_mib) ** 0.6 / max(node.cpu_milli_total, node.memory_mib_total) * 300
    
    util_score = (min(cpu_util, mem_util) * 0.6 + max(cpu_util, mem_util) * 0.4) * 600
    score = util_score + balance_score + gpu_score + frag_score
    
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
            'funsearch_v1': funsearch_v1_scheduler,
            'funsearch_v2': funsearch_v2_scheduler,
            'funsearch_v3': funsearch_v3_scheduler,
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
                print(f"  Average CPU Utilization:  {eval_results.avg_cpu_utilization:.1%}")
                print(f"  Average Memory Utilization: {eval_results.avg_memory_utilization:.1%}")
                print(f"  Average GPU Count Util:   {eval_results.avg_gpu_count_utilization:.1%}")
                print(f"  Average GPU Memory Util:  {eval_results.avg_gpu_memory_utilization:.1%}")
                print(f"  GPU Fragmentation Score:  {eval_results.gpu_fragmentation_score:.3f}")
                print(f"  Utilization Snapshots:    {eval_results.num_snapshots}")
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