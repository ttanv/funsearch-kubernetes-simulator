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
    """Evaluates scheduling policy performance with time-based utilization snapshots and fragmentation tracking"""
    
    def __init__(self, cluster: Cluster, enabled: bool = True, snapshot_interval: float = 0.01):
        self.enabled = enabled
        self.snapshot_interval = snapshot_interval  # Take snapshot every 1% of simulation time by default
        
        # Pre-calculate cluster totals (constant for the experiment)
        self.total_cpu = sum(node.cpu_milli_total for node in cluster.nodes_dict.values())
        self.total_memory = sum(node.memory_mib_total for node in cluster.nodes_dict.values())
        self.total_gpu_count = sum(len(node.gpus) for node in cluster.nodes_dict.values())
        self.total_gpu_memory = sum(gpu.gpu_milli_total for node in cluster.nodes_dict.values() for gpu in node.gpus)
        
        # Time-based tracking data
        self.utilization_snapshots: list[UtilizationSnapshot] = []
        self.fragmentation_events: list[float] = []  # GPU fragmentation scores
        self.total_simulation_time: int = 0
        self.current_time: int = 0
        self.time_since_last_snapshot: float = 0.0
        self.next_snapshot_time: float = 0.0
        self.time_weighted_utilization: dict = {'cpu': 0.0, 'memory': 0.0, 'gpu_count': 0.0, 'gpu_memory': 0.0}
        self.accumulated_time: float = 0.0
        
    def initialize(self, total_simulation_time: int):
        """Initialize evaluator with total simulation time for time-based snapshots"""
        if not self.enabled:
            return
        self.total_simulation_time = total_simulation_time
        self.current_time = 0
        self.time_since_last_snapshot = 0.0
        self.next_snapshot_time = total_simulation_time * self.snapshot_interval
        self.time_weighted_utilization = {'cpu': 0.0, 'memory': 0.0, 'gpu_count': 0.0, 'gpu_memory': 0.0}
        self.accumulated_time = 0.0
        
    def record_event_processed(self, cluster: Cluster, event_time: int):
        """Record event processing with time-based utilization tracking"""
        if not self.enabled:
            return
            
        # Calculate time elapsed since last event
        time_delta = event_time - self.current_time
        
        # Add weighted utilization for the time period that just passed
        if time_delta > 0:
            current_util = self._get_current_utilization(cluster)
            self.time_weighted_utilization['cpu'] += current_util['cpu'] * time_delta
            self.time_weighted_utilization['memory'] += current_util['memory'] * time_delta
            self.time_weighted_utilization['gpu_count'] += current_util['gpu_count'] * time_delta
            self.time_weighted_utilization['gpu_memory'] += current_util['gpu_memory'] * time_delta
            self.accumulated_time += time_delta
        
        self.time_since_last_snapshot += time_delta
        self.current_time = event_time
        
        # Check if we need to take snapshot(s)
        while self.time_since_last_snapshot >= self.next_snapshot_time:
            self._finalize_snapshot()
            self.time_since_last_snapshot -= self.next_snapshot_time
            
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
    
    def get_policy_score(self, pods: list[Pod]) -> float:
        """Calculate a single representative score (0-1) combining all utilization metrics and GPU fragmentation"""
        results = self.get_evaluation_results()
        if not results:
            return 0.0
        
        # Combine all utilization metrics with equal weight
        overall_utilization = (
            results.avg_cpu_utilization + 
            results.avg_memory_utilization + 
            results.avg_gpu_count_utilization + 
            results.avg_gpu_memory_utilization
        ) / 4.0
        
        # Apply fragmentation penalty (bounded to maximum of 0.1)
        fragmentation_penalty = min(0.1, results.gpu_fragmentation_score)
        
        # Final score: overall utilization minus fragmentation penalty
        # Clamp to [0, 1] range
        score = max(0.0, min(1.0, overall_utilization - fragmentation_penalty))
        
        return score
    
    def _get_current_utilization(self, cluster: Cluster) -> dict:
        """Get current instantaneous utilization"""
        used_cpu = sum(node.cpu_milli_total - node.cpu_milli_left for node in cluster.nodes_dict.values())
        used_memory = sum(node.memory_mib_total - node.memory_mib_left for node in cluster.nodes_dict.values())
        used_gpu_count = sum(len(node.gpus) - node.gpu_left for node in cluster.nodes_dict.values())
        used_gpu_memory = sum(gpu.gpu_milli_total - gpu.gpu_milli_left for node in cluster.nodes_dict.values() for gpu in node.gpus)
        
        return {
            'cpu': used_cpu / self.total_cpu if self.total_cpu > 0 else 0.0,
            'memory': used_memory / self.total_memory if self.total_memory > 0 else 0.0,
            'gpu_count': used_gpu_count / self.total_gpu_count if self.total_gpu_count > 0 else 0.0,
            'gpu_memory': used_gpu_memory / self.total_gpu_memory if self.total_gpu_memory > 0 else 0.0
        }
    
    def _finalize_snapshot(self):
        """Create a snapshot from accumulated time-weighted utilization"""
        if self.accumulated_time <= 0:
            return
            
        # Calculate time-weighted average utilization for this snapshot period
        snapshot = UtilizationSnapshot(
            cpu_utilization=self.time_weighted_utilization['cpu'] / self.accumulated_time,
            memory_utilization=self.time_weighted_utilization['memory'] / self.accumulated_time,
            gpu_count_utilization=self.time_weighted_utilization['gpu_count'] / self.accumulated_time,
            gpu_memory_utilization=self.time_weighted_utilization['gpu_memory'] / self.accumulated_time,
            event_progress=len(self.utilization_snapshots) * self.snapshot_interval  # Time progress
        )
        
        self.utilization_snapshots.append(snapshot)
        
        # Reset accumulators for next snapshot period
        self.time_weighted_utilization = {'cpu': 0.0, 'memory': 0.0, 'gpu_count': 0.0, 'gpu_memory': 0.0}
        self.accumulated_time = 0.0
    
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