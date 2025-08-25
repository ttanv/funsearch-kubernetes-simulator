from dataclasses import dataclass
from typing import Optional
import statistics

from simulator.entities import Cluster, Pod

@dataclass
class EvaluationResults:
    """Final evaluation results for a scheduling policy"""
    gpu_fragmentation_score: float
    num_fragmentation_events: int
    

    total_scheduling_attempts: int = 0
    peak_nodes_used: int = 0
    total_repush_events: int = 0

class SchedulingEvaluator:
    """Evaluates scheduling policy performance with focus on scheduling efficiency and wait times"""
    
    def __init__(self, cluster: Cluster, enabled: bool = True, snapshot_interval: float = 0.01):
        self.enabled = enabled
        self.snapshot_interval = snapshot_interval
        self.cluster = cluster
        
        # Pre-calculate cluster totals (constant for the experiment)
        self.total_gpu_memory = sum(gpu.gpu_milli_total for node in cluster.nodes_dict.values() for gpu in node.gpus)
        self.total_nodes = len(cluster.nodes_dict)
        
        self.fragmentation_events: list[float] = []
        self.current_time: int = 0
        self.peak_nodes_used: int = 0
        
        # Scheduling attempts tracking
        self.total_scheduling_attempts: int = 0
        self.pod_repush_counts: dict[str, int] = {}
        
    def initialize(self, time):
        """Initialize evaluator with total simulation time for time-based snapshots"""
        if not self.enabled:
            return
        
    def record_event_processed(self, cluster: Cluster, event_time: int):
        """Record event processing with enhanced metrics tracking"""
        if not self.enabled:
            return
        
        # Update node usage metrics
        self._update_node_usage_metrics(cluster)
        
        self.current_time = event_time
        
    def record_fragmentation_event(self, cluster: Cluster, waiting_pods: list[Pod]):
        """Record when pods fail to schedule and must be repushed"""
        if not self.enabled:
            return
        
        # Process each waiting pod
        for pod in waiting_pods:
            pod_id = pod.pod_id

            self.total_scheduling_attempts += 1
            if pod_id not in self.pod_repush_counts:
                self.pod_repush_counts[pod_id] = 1
            else:
                # This is a repush
                self.pod_repush_counts[pod_id] += 1
        
        # Calculate fragmentation score 
        if waiting_pods:
            fragmentation_score = self._calculate_gpu_fragmentation(cluster, waiting_pods)
            self.fragmentation_events.append(fragmentation_score)
            
    def record_successful_scheduling(self, pod: Pod):
        """Called when a pod is successfully scheduled """
        if not self.enabled:
            return
            
        pod_id = pod.pod_id
        
        # Track scheduling attempt
        self.total_scheduling_attempts += 1
        
        # Initialize if this is the first time we see this pod
        if pod_id not in self.pod_repush_counts:
            self.pod_repush_counts[pod_id] = 0
        
    def _update_node_usage_metrics(self, cluster: Cluster):
        """Track node usage over time"""
        # Identify currently active nodes
        active_nodes = set()
        for node_id, node in cluster.nodes_dict.items():
            if (node.cpu_milli_left < node.cpu_milli_total or 
                node.memory_mib_left < node.memory_mib_total or 
                node.gpu_left < len(node.gpus)):
                active_nodes.add(node_id)
        
        num_active = len(active_nodes)
        
        # Update peak
        self.peak_nodes_used = max(self.peak_nodes_used, num_active)
        
    def _calculate_gpu_fragmentation(self, cluster: Cluster, waiting_pods: list[Pod]) -> float:
        """Calculate GPU fragmentation score"""
        if not waiting_pods:
            return 0.0
            
        gpu_waiting_pods = [pod for pod in waiting_pods if pod.num_gpu > 0]
        if not gpu_waiting_pods:
            return 0.0
            
        min_gpu_milli_needed = min(pod.gpu_milli for pod in gpu_waiting_pods)
        fragmented_gpu_memory = 0
        
        for node in cluster.nodes_dict.values():
            for gpu in node.gpus:
                if 0 < gpu.gpu_milli_left < min_gpu_milli_needed:
                    fragmented_gpu_memory += gpu.gpu_milli_left
                    
        return fragmented_gpu_memory / self.total_gpu_memory if self.total_gpu_memory > 0 else 0.0
        
    def get_evaluation_results(self) -> Optional[EvaluationResults]:
        """Calculate and return final evaluation results"""
        if not self.enabled:
            return None
            
        avg_fragmentation = statistics.mean(self.fragmentation_events) if self.fragmentation_events else 0.0
            
        total_repushes = sum(self.pod_repush_counts.values())
        
        return EvaluationResults(
            gpu_fragmentation_score=avg_fragmentation,
            num_fragmentation_events=len(self.fragmentation_events),
            
            total_scheduling_attempts=self.total_scheduling_attempts,
            
            peak_nodes_used=self.peak_nodes_used,
            
            total_repush_events=total_repushes,
        )
    
    def get_policy_score(self, pods: list[Pod]) -> float:
        """Calculate a single score optimized for FunSearch/AlphaEvolve"""
        results = self.get_evaluation_results()
        if not results:
            return 0.0
        
        if self.total_scheduling_attempts > 0:
            repush_rate = results.total_repush_events / self.total_scheduling_attempts
            success_score = 1.0 / (1.0 + repush_rate)
        else:
            success_score = 0.0
            
        if self.total_nodes > 0:
            node_efficiency = 1.0 - (results.peak_nodes_used / self.total_nodes)
        else:
            node_efficiency = 0.0
            
        avg_repushes_per_pod = results.total_repush_events / max(1, len(pods))
        wait_penalty = min(1.0, avg_repushes_per_pod )
        
        frag_penalty = min(0.2, results.gpu_fragmentation_score)
        
        score = (
            0.40 * success_score +
            0.30 * node_efficiency +
            0.50 * (1 - wait_penalty) +
            0.10 * (1 - frag_penalty)
        )
        
        return score