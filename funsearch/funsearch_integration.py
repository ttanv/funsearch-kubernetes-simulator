"""
Integration of FunSearch with the Kubernetes simulator
Uses the full OpenB dataset evaluation from test_scheduler.py
"""

import openai
import json
import os
import sys
import random
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Callable
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from funsearch.safe_execution import SafeExecutor, LLMCodeGenerator
from benchmarks.parser import TraceParser
from simulator.entities import Cluster, Node, Pod
from simulator.event_simulator import DiscreteEventSimulator
from simulator.main import KubernetesSimulator
from simulator.evaluator import SchedulingEvaluator


class FunSearchScheduler:
    """FunSearch-evolved scheduler that integrates with the simulator"""
    
    def __init__(self, evolved_code: str, safe_executor: SafeExecutor):
        self.evolved_code = evolved_code
        self.safe_executor = safe_executor
        self._compiled_function = None
        self._compile_policy()
        self.fallback_print = True
    
    def _compile_policy(self):
        """Pre-compile the policy function for faster execution"""
        try:
            # Create safe environment once
            safe_env = self.safe_executor.create_safe_environment()
            exec(self.evolved_code, safe_env)
            self._compiled_function = safe_env.get('priority_function')
            
            if not self._compiled_function:
                raise ValueError("No priority_function found in evolved code")
                
        except Exception as e:
            raise ValueError(f"Failed to compile evolved policy: {e}")
    
    def __call__(self, pod: Pod, node: Node) -> int:
        """Scheduler interface that matches the existing system"""
        try:
            # Call the evolved function with correct signature (pod, node)
            score = self._compiled_function(pod, node)
            return int(max(0, score))  # Ensure non-negative integer
            
        except Exception as e:
            # Fallback if evolved policy fails
            if self.fallback_print:
                print(f"Evolved policy failed: {e}")
                self.fallback_print = False
                
            return 0
    
    def _fallback_score(self, pod: Pod, node: Node) -> int:
        """Safe fallback scoring if evolved policy fails"""
        # Simple logic as fallback
        if (pod.cpu_milli <= node.cpu_milli_left and 
            pod.memory_mib <= node.memory_mib_left and
            pod.num_gpu <= node.gpu_left):
            
            # Check GPU requirements
            if pod.num_gpu > 0:
                available_gpus = 0
                for gpu in node.gpus:
                    if gpu.gpu_milli_left >= pod.gpu_milli:
                        available_gpus += 1
                if available_gpus < pod.num_gpu:
                    return 0
            
            return 1000 - node.cpu_milli_left  
        return 0  


class SimpleFunSearch:
    """Simplified FunSearch implementation using full OpenB evaluation"""
    
    def __init__(self, config_path: str = "configs/llm_config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.safe_executor = SafeExecutor(
            timeout_seconds=self.config["safe_execution"]["timeout_seconds"]
        )
        
        # Initialize OpenRouter client
        openrouter_config = self.config["openrouter"]
        self.llm_client = openai.OpenAI(
            api_key=openrouter_config["api_key"],
            base_url=openrouter_config["base_url"]
        )
        self.model = openrouter_config["model"]
        max_tokens = openrouter_config.get("max_tokens", 400)
        self.code_generator = LLMCodeGenerator(self.llm_client, self.safe_executor, self.model, max_tokens)
        
        # Evolution parameters
        funsearch_config = self.config["funsearch"]
        self.population_size = funsearch_config["population_size"]
        self.max_generations = funsearch_config["generations"]
        self.early_stop_threshold = funsearch_config["early_stop_threshold"]
        self.elite_size = funsearch_config["elite_size"]
        
        # Load workload data once
        print("Loading OpenB dataset...")
        parser = TraceParser()
        self.original_cluster, self.original_pods = parser.parse_workload()
        print(f"Loaded {len(self.original_cluster.nodes_dict)} nodes and {len(self.original_pods)} pods")
        
        # Evolution state
        self.population = []  # List of (code, score) tuples
        self.generation = 0
        self.best_policy = None
        self.best_score = float('-inf')
    
    def initialize_population(self) -> None:
        """Initialize population with baseline policies"""
        baseline_policies = [
            self._create_first_fit_policy(),
            # self._create_best_fit_policy(),
            # self._create_worst_fit_policy(),
            # self._create_gpu_aware_policy(),
            # self._create_utilization_based_policy(),
        ]
        
        # Add some random variations
        for _ in range(3):
            baseline_policies.append(self._create_random_policy())
        
        print("Evaluating baseline policies on OpenB dataset...")
        
        # Evaluate baseline policies using full simulation
        for i, policy_code in enumerate(baseline_policies):
            print(f"Evaluating baseline policy {i+1}/{len(baseline_policies)}...", end=" ")
            score = self._evaluate_policy_full(policy_code)
            if score is not None:
                self.population.append((policy_code, score))
                if score > self.best_score:
                    self.best_score = score
                    self.best_policy = policy_code
            print(f"Score: {score:.4f}" if score else "Failed")
        
        # Keep only the best policies
        self.population.sort(key=lambda x: x[1], reverse=True)
        self.population = self.population[:self.population_size]
        
        print(f"Initialized population with {len(self.population)} policies")
        print(f"Best baseline score: {self.best_score:.4f}")
    
    def _create_first_fit_policy(self) -> str:
        """Create first-fit policy"""
        return '''
def priority_function(pod, node):
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
'''
    
    def _create_best_fit_policy(self) -> str:
        """Create best-fit policy"""
        return '''
def priority_function(pod, node):
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
'''
    
    def _create_worst_fit_policy(self) -> str:
        """Create worst-fit policy"""
        return '''
def priority_function(pod, node):
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
'''
    
    def _create_gpu_aware_policy(self) -> str:
        """Create GPU-aware policy (workload_aware from test_scheduler.py)"""
        return '''
def priority_function(pod, node):
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
'''
    
    def _create_utilization_based_policy(self) -> str:
        """Create utilization-based policy (hybrid from test_scheduler.py)"""
        return '''
def priority_function(pod, node):
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
'''
    
    def _create_random_policy(self) -> str:
        """Create randomized policy"""
        cpu_factor = random.uniform(0.0001, 0.01)
        mem_factor = random.uniform(0.00001, 0.001)
        gpu_factor = random.uniform(10.0, 1000.0)
        base_score = random.uniform(1000.0, 5000.0)
        
        return f'''
def priority_function(pod, node):
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
    
    score = {base_score} + node.cpu_milli_left * {cpu_factor} + node.memory_mib_left * {mem_factor}
    
    if pod.num_gpu > 0 and node.gpu_left > 0:
        score += node.gpu_left * {gpu_factor}
    
    return max(1, int(score))
'''
    
    def _evaluate_policy_full(self, policy_code: str) -> Optional[float]:
        """Evaluate a policy using the full OpenB simulation"""
        try:
            # Create FunSearch scheduler
            scheduler = FunSearchScheduler(policy_code, self.safe_executor)
            
            # Run full simulation using the same method as test_scheduler.py
            import copy
            cluster = copy.deepcopy(self.original_cluster)
            pods = copy.deepcopy(self.original_pods)
            
            # Initialize simulator with evaluator
            event_simulator = DiscreteEventSimulator(pods)
            evaluator = SchedulingEvaluator(cluster, enabled=True)
            simulator = KubernetesSimulator(cluster, pods, event_simulator, scheduler, evaluator=evaluator)
            
            # Run simulation
            simulator.run_schedule()
            
            # Get policy score (0-1 range)
            policy_score = simulator.evaluator.get_policy_score()
            
            return policy_score
            
        except Exception as e:
            print(f"Failed to evaluate policy: {e}")
            return None
    
    def evolve_generation(self):
        """Evolve one generation using FunSearch"""
        self.generation += 1
        new_policies = []
        
        print(f"\n--- Generation {self.generation} ---")
        
        # Select top performers
        self.population.sort(key=lambda x: x[1], reverse=True)
        elite_policies = self.population[:self.elite_size]
        
        # Generate new policies
        policies_to_generate = min(8, self.population_size - len(elite_policies))
        
        for i in range(policies_to_generate):
            print(f"Generating policy {i+1}/{policies_to_generate}...", end=" ")
            
            # Select parent policies
            parents = random.sample(elite_policies, min(2, len(elite_policies)))
            
            # Create performance feedback
            feedback = f"Best score so far: {self.best_score:.4f}. "
            feedback += f"Elite policies achieve good performance by balancing resource utilization "
            feedback += f"and considering GPU/CPU workload separation. "
            feedback += f"Focus on: CPU/memory efficiency, GPU placement strategies, fragmentation reduction."
            
            # Generate new policy
            new_code = self.code_generator.generate_policy(
                parent_policies=parents,
                performance_feedback=feedback
            )
            
            if new_code:
                # Evaluate new policy
                score = self._evaluate_policy_full(new_code)
                if score is not None:
                    new_policies.append((new_code, score))
                    
                    # Update best if improved
                    if score > self.best_score:
                        self.best_score = score
                        self.best_policy = new_code
                        print(f"NEW BEST! Score: {score:.4f}")
                    else:
                        print(f"Score: {score:.4f}")
                else:
                    print("Failed")
            else:
                print("Generation failed")
        
        # Combine elite and new policies
        all_policies = elite_policies + new_policies
        
        # Keep population size manageable
        all_policies.sort(key=lambda x: x[1], reverse=True)
        self.population = all_policies[:self.population_size]
        
        print(f"Population: {len(self.population)} policies, "
              f"best score: {self.best_score:.4f}")
    
    def run_evolution(self, generations: Optional[int] = None) -> Tuple[str, float]:
        """Run complete FunSearch evolution"""
        generations = generations or self.max_generations
        
        print("Starting FunSearch evolution...")
        
        # Initialize if not done
        if not self.population:
            self.initialize_population()
        
        # Evolve for specified generations
        for gen in range(generations):
            start_time = time.time()
            self.evolve_generation()
            gen_time = time.time() - start_time
            print(f"Generation {self.generation} completed in {gen_time:.1f}s")
            
            # Early stopping if we find a really good policy or overfit?
            if self.best_score >= self.early_stop_threshold:
                print(f"Reached target score ({self.best_score:.4f}), stopping early")
                break
        
        print(f"Evolution complete! Best score: {self.best_score:.4f}")
        return self.best_policy, self.best_score
    
    def get_best_scheduler(self) -> FunSearchScheduler:
        """Get the best evolved scheduler for use in the simulator"""
        if not self.best_policy:
            raise ValueError("No evolved policy available. Run evolution first.")
        
        return FunSearchScheduler(self.best_policy, self.safe_executor)
    
    def save_best_policy(self, filepath: Optional[str] = None) -> str:
        """Save the best evolved policy to a file with timestamp"""
        if not self.best_policy:
            raise ValueError("No best policy to save")
        
        # Generate timestamped filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("policies/discovered", exist_ok=True)
            filepath = f"policies/discovered/funsearch_{timestamp}_score{self.best_score:.4f}.json"
        else:
            # If filepath is provided, still add timestamp to avoid overwriting
            base, ext = os.path.splitext(filepath)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{base}_{timestamp}{ext}"
        
        policy_data = {
            "score": self.best_score,
            "generation": self.generation,
            "code": self.best_policy,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(policy_data, f, indent=2)
        
        print(f"Best policy saved to {filepath}")
        return filepath


def main():
    """Example usage of FunSearch integration"""
    # Initialize FunSearch
    funsearch = SimpleFunSearch()
    
    # Run evolution
    try:
        best_policy, best_score = funsearch.run_evolution(generations=3)  # Start with 3 generations
        
        # Save the best policy
        saved_filepath = funsearch.save_best_policy()
        
        print(f"\nFinal Results:")
        print(f"Best Score: {best_score:.4f}")
        print(f"Policy saved to {saved_filepath}")
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        if funsearch.best_policy:
            interrupted_filepath = funsearch.save_best_policy()
            print(f"Current best policy saved to {interrupted_filepath}")


if __name__ == "__main__":
    main()