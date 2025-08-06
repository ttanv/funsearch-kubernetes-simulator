from dataclasses import dataclass
from typing import Optional
import statistics

from simulator.entities import Cluster, Pod

@dataclass
class UtilizationSnapshot:
    """Represents resource utilization at a point in time"""
    cpu_utilization: float
    memory_utilization: float
    gpu_count_utilization: float
    gpu_memory_utilization: float
    event_progress: float

@dataclass
class EvaluationResults:
    """Final evaluation results for a scheduling policy"""
    avg_cpu_utilization: float
    avg_memory_utilization: float
    avg_gpu_count_utilization: float
    avg_gpu_memory_utilization: float
    gpu_fragmentation_score: float
    num_snapshots: int
    num_fragmentation_events: int

class SchedulingEvaluator:
    """Evaluates scheduling policy performance with utilization snapshots and fragmentation tracking"""
    
    def __init__(self, cluster: Cluster, enabled: bool = True, snapshot_interval: float = 0.05):
        self.enabled = enabled
        self.snapshot_interval = snapshot_interval  # Take snapshot every 5% of events by default
        
        # Pre-calculate cluster totals (constant for the experiment)
        self.total_cpu = sum(node.cpu_milli_total for node in cluster.nodes_dict.values())
        self.total_memory = sum(node.memory_mib_total for node in cluster.nodes_dict.values())
        self.total_gpu_count = sum(len(node.gpus) for node in cluster.nodes_dict.values())
        self.total_gpu_memory = sum(gpu.gpu_milli_total for node in cluster.nodes_dict.values() for gpu in node.gpus)
        
        # Tracking data
        self.utilization_snapshots: list[UtilizationSnapshot] = []
        self.fragmentation_events: list[float] = []  # GPU fragmentation scores
        self.total_events: int = 0
        self.events_processed: int = 0
        self.next_snapshot_threshold: float = snapshot_interval
        
    def initialize(self, total_events: int):
        """Initialize evaluator with total number of events for progress tracking"""
        if not self.enabled:
            return
        self.total_events = total_events
        self.events_processed = 0
        self.next_snapshot_threshold = self.snapshot_interval
        
    def record_event_processed(self, cluster: Cluster):
        """Record that an event was processed and take snapshot if needed"""
        if not self.enabled:
            return
            
        self.events_processed += 1
        progress = self.events_processed / self.total_events if self.total_events > 0 else 0
        
        # Take snapshot if we've crossed the threshold
        if progress >= self.next_snapshot_threshold:
            snapshot = self._take_utilization_snapshot(cluster, progress)
            self.utilization_snapshots.append(snapshot)
            self.next_snapshot_threshold += self.snapshot_interval
            
    def record_fragmentation_event(self, cluster: Cluster, waiting_pods: list[Pod]):
        """Record GPU fragmentation when pods are repushed due to insufficient resources"""
        if not self.enabled or not waiting_pods:
            return
            
        fragmentation_score = self._calculate_gpu_fragmentation(cluster, waiting_pods)
        self.fragmentation_events.append(fragmentation_score)
        
    def get_evaluation_results(self) -> Optional[EvaluationResults]:
        """Calculate and return final evaluation results"""
        if not self.enabled or not self.utilization_snapshots:
            return None
            
        # Calculate average utilization across all snapshots
        avg_cpu = statistics.mean(s.cpu_utilization for s in self.utilization_snapshots)
        avg_memory = statistics.mean(s.memory_utilization for s in self.utilization_snapshots)
        avg_gpu_count = statistics.mean(s.gpu_count_utilization for s in self.utilization_snapshots)
        avg_gpu_memory = statistics.mean(s.gpu_memory_utilization for s in self.utilization_snapshots)
        
        # Calculate average fragmentation score
        avg_fragmentation = statistics.mean(self.fragmentation_events) if self.fragmentation_events else 0.0
        
        return EvaluationResults(
            avg_cpu_utilization=avg_cpu,
            avg_memory_utilization=avg_memory,
            avg_gpu_count_utilization=avg_gpu_count,
            avg_gpu_memory_utilization=avg_gpu_memory,
            gpu_fragmentation_score=avg_fragmentation,
            num_snapshots=len(self.utilization_snapshots),
            num_fragmentation_events=len(self.fragmentation_events)
        )
    
    def _take_utilization_snapshot(self, cluster: Cluster, progress: float) -> UtilizationSnapshot:
        """Take a snapshot of current cluster utilization"""
        used_cpu = sum(node.cpu_milli_total - node.cpu_milli_left for node in cluster.nodes_dict.values())
        used_memory = sum(node.memory_mib_total - node.memory_mib_left for node in cluster.nodes_dict.values())
        used_gpu_count = sum(len(node.gpus) - node.gpu_left for node in cluster.nodes_dict.values())
        used_gpu_memory = sum(gpu.gpu_milli_total - gpu.gpu_milli_left for node in cluster.nodes_dict.values() for gpu in node.gpus)
        
        return UtilizationSnapshot(
            cpu_utilization=used_cpu / self.total_cpu if self.total_cpu > 0 else 0.0,
            memory_utilization=used_memory / self.total_memory if self.total_memory > 0 else 0.0,
            gpu_count_utilization=used_gpu_count / self.total_gpu_count if self.total_gpu_count > 0 else 0.0,
            gpu_memory_utilization=used_gpu_memory / self.total_gpu_memory if self.total_gpu_memory > 0 else 0.0,
            event_progress=progress
        )
    
    def _calculate_gpu_fragmentation(self, cluster: Cluster, waiting_pods: list[Pod]) -> float:
        """Calculate GPU fragmentation score based on unusable GPU memory"""
        if not waiting_pods:
            return 0.0
            
        # Find minimum GPU memory requirement among waiting pods
        gpu_waiting_pods = [pod for pod in waiting_pods if pod.num_gpu > 0]
        if not gpu_waiting_pods:
            return 0.0
            
        min_gpu_milli_needed = min(pod.gpu_milli for pod in gpu_waiting_pods)
        
        # Count fragmented GPU memory (available but too small for any waiting pod)
        fragmented_gpu_memory = 0
        
        for node in cluster.nodes_dict.values():
            for gpu in node.gpus:
                if 0 < gpu.gpu_milli_left < min_gpu_milli_needed:
                    fragmented_gpu_memory += gpu.gpu_milli_left
                    
        return fragmented_gpu_memory / self.total_gpu_memory if self.total_gpu_memory > 0 else 0.0