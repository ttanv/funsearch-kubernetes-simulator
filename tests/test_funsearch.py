#!/usr/bin/env python3

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from funsearch.safe_execution import SafeExecutor, PolicyTemplate, LLMCodeGenerator
from funsearch.funsearch_integration import FunSearchScheduler, SimpleFunSearch
from simulator.entities import Node, Pod, GPU


class TestSafeExecution(unittest.TestCase):
    
    def setUp(self):
        self.executor = SafeExecutor(timeout_seconds=1)
        
    def test_safe_code_execution(self):
        """Test that safe code executes properly"""
        safe_code = '''
def priority_function(pod, node):
    score = 0.0
    if node.cpu_milli_left >= pod.cpu_milli:
        score = 100.0
    return score
'''
        
        # Mock objects
        pod = Mock()
        pod.cpu_milli = 100
        node = Mock()
        node.cpu_milli_left = 200
        
        result = self.executor.execute_policy_function(safe_code, pod, node)
        self.assertEqual(result, 100.0)
    
    def test_unsafe_code_blocked(self):
        """Test that unsafe code is blocked"""
        unsafe_code = '''
import os
def priority_function(pod, node):
    os.system("echo danger")
    return 1.0
'''
        
        pod = Mock()
        node = Mock()
        
        with self.assertRaises(ValueError):
            self.executor.execute_policy_function(unsafe_code, pod, node)
    
    def test_template_system(self):
        """Test the policy template system"""
        llm_logic = '''
if node.cpu_milli_left > pod.cpu_milli * 2:
    score = 1000.0
else:
    score = 500.0
'''
        
        complete_code = PolicyTemplate.fill_template(llm_logic)
        self.assertIn('def priority_function', complete_code)
        self.assertIn(llm_logic.strip(), complete_code)


class TestFunSearchScheduler(unittest.TestCase):
    
    def setUp(self):
        self.executor = SafeExecutor()
        
        # Create a simple policy
        self.policy_code = '''
def priority_function(pod, node):
    score = 0.0
    if (node.cpu_milli_left >= pod.cpu_milli and 
        node.memory_mib_left >= pod.memory_mib and
        node.gpu_left >= pod.num_gpu):
        # Best fit: higher score for less remaining capacity
        remaining_ratio = node.cpu_milli_left / node.cpu_milli_total
        score = 1000.0 * (1.0 - remaining_ratio)  # 0-1000 range
    return score
'''
    
    def test_scheduler_creation(self):
        """Test creating a FunSearch scheduler"""
        scheduler = FunSearchScheduler(self.policy_code, self.executor)
        self.assertIsNotNone(scheduler._compiled_function)
    
    def test_scheduler_call(self):
        """Test calling the scheduler with pod/node"""
        scheduler = FunSearchScheduler(self.policy_code, self.executor)
        
        # Create test entities matching simulator
        gpu = GPU(memory_mib_left=8000, memory_mib_total=8000, 
                 gpu_milli_left=1000, gpu_milli_total=1000)
        
        node = Node(node_id="test-node", cpu_milli_left=2000, cpu_milli_total=4000,
                   memory_mib_left=4000, memory_mib_total=8000, 
                   gpu_left=1, gpus=[gpu])
        
        pod = Pod(pod_id="test-pod", cpu_milli=1000, memory_mib=2000,
                 num_gpu=0, gpu_milli=0, gpu_spec="", 
                 creation_time=0, duration_time=100,
                 assigned_node="", assigned_gpus=[])
        
        score = scheduler(pod, node)
        self.assertIsInstance(score, (int, float))
        self.assertGreater(score, 0)
    
    def test_scheduler_fallback(self):
        """Test scheduler fallback on policy failure"""
        bad_policy = '''
def priority_function(pod, node):
    return 1 / 0  # This will cause an error
'''

        print(PolicyTemplate.create_prompt_for_llm([(self.policy_code, 0.5), (bad_policy, 0.1)], "Improve"))
        
        scheduler = FunSearchScheduler(bad_policy, self.executor)
        
        # Create test entities
        gpu = GPU(memory_mib_left=8000, memory_mib_total=8000, 
                 gpu_milli_left=1000, gpu_milli_total=1000)
        
        node = Node(node_id="test-node", cpu_milli_left=2000, cpu_milli_total=4000,
                   memory_mib_left=4000, memory_mib_total=8000, 
                   gpu_left=1, gpus=[gpu])
        
        pod = Pod(pod_id="test-pod", cpu_milli=1000, memory_mib=2000,
                 num_gpu=0, gpu_milli=0, gpu_spec="", 
                 creation_time=0, duration_time=100,
                 assigned_node="", assigned_gpus=[])
        
        # Should fallback gracefully
        score = scheduler(pod, node)
        self.assertIsInstance(score, int)


class TestFunSearchIntegration(unittest.TestCase):
    
    @patch('funsearch.funsearch_integration.TraceParser')
    @patch('builtins.open')
    def test_funsearch_initialization(self, mock_open, mock_parser):
        """Test FunSearch initialization without real data"""
        # Mock the config file
        mock_config = {
            "openai": {"api_key": "test-key"},
            "openrouter": {
                "api_key": "test-key",
                "base_url": "http://localhost:9702/v1",
                "model": "qwen3-coder",
                "max_tokens": 1024,
                "temperature": 0.7
            },
            "safe_execution": {"timeout_seconds": 3},
            "funsearch": {
                "population_size": 5,
                "generations": 2,
                "early_stop_threshold": 0.95,
                "elite_size": 2
            }
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = str(mock_config)
        
        # Mock the parser
        mock_cluster = Mock()
        mock_cluster.nodes_dict = {"node1": Mock(), "node2": Mock()}
        mock_pods = [Mock() for _ in range(10)]
        mock_parser.return_value.parse_workload.return_value = (mock_cluster, mock_pods)
        
        with patch('json.load', return_value=mock_config):
            with patch('openai.OpenAI'):
                funsearch = SimpleFunSearch("test_config.json")
                
                self.assertEqual(funsearch.population_size, 5)
                self.assertEqual(funsearch.max_generations, 2)
                self.assertIsNotNone(funsearch.safe_executor)


def run_basic_tests():
    """Run basic tests that don't require OpenAI API or full dataset"""
    print("Running basic FunSearch integration tests...")
    
    # Test safe executor
    print("Testing SafeExecutor...")
    executor = SafeExecutor(timeout_seconds=1)
    
    template = PolicyTemplate()
    
    test_code = '''
def priority_function(pod, node):
    if node.cpu_milli_left >= pod.cpu_milli:
        return 1000.0
    return 0.0
'''
    
    # Mock objects
    pod = Mock()
    pod.cpu_milli = 100
    node = Mock()
    node.cpu_milli_left = 200
    
    try:
        result = executor.execute_policy_function(test_code, pod, node)
        print(f"Safe execution works: {result}")
    except Exception as e:
        print(f"Safe execution failed: {e}")
        return False
    
    # Test unsafe code blocking
    print("Testing unsafe code blocking...")
    unsafe_code = '''
import subprocess
def priority_function(pod, node):
    return 1.0
'''
    
    try:
        executor.execute_policy_function(unsafe_code, pod, node)
        print("Unsafe code was not blocked!")
        return False
    except ValueError:
        print("Unsafe code properly blocked")
    
    # Test FunSearchScheduler
    print("Testing FunSearchScheduler...")
    try:
        scheduler = FunSearchScheduler(test_code, executor)
        
        # Create real entities for testing
        gpu = GPU(memory_mib_left=8000, memory_mib_total=8000, 
                 gpu_milli_left=1000, gpu_milli_total=1000)
        
        node = Node(node_id="test-node", cpu_milli_left=2000, cpu_milli_total=4000,
                   memory_mib_left=4000, memory_mib_total=8000, 
                   gpu_left=1, gpus=[gpu])
        
        pod = Pod(pod_id="test-pod", cpu_milli=1000, memory_mib=2000,
                 num_gpu=0, gpu_milli=0, gpu_spec="", 
                 creation_time=0, duration_time=100,
                 assigned_node="", assigned_gpus=[])
        
        score = scheduler(pod, node)
        print(f"Scheduler works: {score}")
        
    except Exception as e:
        print(f"Scheduler failed: {e}")
        return False
    
    print("All basic tests passed!")
    return True


if __name__ == "__main__":
    unittest.main()