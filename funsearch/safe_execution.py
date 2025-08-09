"""
Safe execution of LLM-generated scheduling policies
Combines multiple safety layers for demo use
"""

import ast
import sys
import signal
import traceback
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
import math
import operator

class SafeExecutor:
    """Safely check LLM-generated scheduling policy functions"""
    
    # Probably not an exhaustive list, works for now
    ALLOWED_BUILTINS = {
        'abs', 'min', 'max', 'sum', 'len', 'range', 'enumerate',
        'int', 'float', 'bool', 'str', 'round'
    }
    
    ALLOWED_MODULES = {
        'math': ['sqrt', 'log', 'exp', 'pow', 'sin', 'cos', 'tan'],
        'operator': ['add', 'sub', 'mul', 'truediv', 'mod']
    }
    
    FORBIDDEN_PATTERNS = [
        'import', '__', 'exec', 'eval', 'open', 'file', 'input', 
        'raw_input', 'compile', 'globals', 'locals', 'vars',
        'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
    ]
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
    
    def validate_code_structure(self, code: str) -> bool:
        """Validate code using AST parsing"""
        try:
            tree = ast.parse(code)
            
            # Check for forbidden constructs
            for node in ast.walk(tree):
                # No imports allowed
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Import statements not allowed")
                
                # No attribute access to dangerous items
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith('__'):
                        raise ValueError(f"Access to {node.attr} not allowed")
                
                # No function calls to dangerous functions
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in self.ALLOWED_BUILTINS:
                            if not self._is_allowed_function_call(node.func.id):
                                raise ValueError(f"Function {node.func.id} not allowed")
            
            return True
            
        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}")
    
    def _is_allowed_function_call(self, func_name: str) -> bool:
        """Check if function call is in whitelist"""
        for _, functions in self.ALLOWED_MODULES.items():
            if func_name in functions:
                return True
        return False
    
    def validate_code_content(self, code: str) -> bool:
        """Basic string validation for forbidden patterns"""
        code_lower = code.lower()
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code_lower:
                raise ValueError(f"Forbidden pattern '{pattern}' found in code")
        return True
    
    @contextmanager
    def timeout_handler(self, seconds: int):
        """Context manager for execution timeout"""
        def timeout_signal_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {seconds} seconds")
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore old handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def create_safe_environment(self) -> Dict[str, Any]:
        """Create restricted execution environment"""
        import builtins
        
        # Create a proper builtins dict
        safe_builtins = {}
        for name in self.ALLOWED_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)
        
        safe_env = {
            # Safe builtins
            '__builtins__': safe_builtins,
            
            # Safe math functions
            'math': type('SafeMath', (), {
                name: getattr(math, name) 
                for name in self.ALLOWED_MODULES['math']
            })(),
            
            # Safe operators
            'operator': type('SafeOperator', (), {
                name: getattr(operator, name) 
                for name in self.ALLOWED_MODULES['operator']
            })(),
        }
        return safe_env
    
    def execute_policy_function(self, code: str, pod, node) -> float:
        """Safely execute a scheduling policy function"""
        
        # Step 1: Validate code structure and content
        self.validate_code_content(code)
        self.validate_code_structure(code)
        
        # Step 2: Create safe execution environment
        safe_env = self.create_safe_environment()
        
        # Step 3: Add function parameters to environment
        safe_env.update({
            'pod': pod,
            'node': node
        })
        
        try:
            # Step 4: Execute with timeout
            with self.timeout_handler(self.timeout_seconds):
                # Execute the code in safe environment
                exec(code, safe_env)
                
                # The code should define a function called 'priority_function'
                if 'priority_function' not in safe_env:
                    raise ValueError("Generated code must define 'priority_function'")
                
                # Call the function with correct signature
                result = safe_env['priority_function'](pod, node)
                
                # Validate result
                if not isinstance(result, (int, float)):
                    raise ValueError(f"Priority function must return a number, got {type(result)}")
                
                # Check for NaN or infinite values
                if math.isnan(result) or math.isinf(result):
                    raise ValueError("Priority function returned NaN or infinite value")
                
                return float(result)
                
        except TimeoutError:
            raise ValueError("Code execution timed out")
        except Exception as e:
            raise ValueError(f"Error executing generated code: {str(e)}")


class PolicyTemplate:
    """Template system to constrain what LLM can generate"""
    
    TEMPLATE = '''
def priority_function(pod, node):
    """
    Calculate priority score for placing pod on node.
    Higher score = better placement.
    
    Available attributes:
    - pod.cpu_milli, pod.memory_mib, pod.num_gpu, pod.gpu_milli
    - node.cpu_milli_left, node.memory_mib_left, node.gpu_left, node.gpus
    - node.cpu_milli_total, node.memory_mib_total
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
    
    # LLM fills in this part
    score = 0.0
    
    {llm_generated_logic}
    
    return max(1, int(score))
'''
    
    @classmethod
    def create_prompt_for_llm(cls, parent_policies: list, performance_feedback: str) -> str:
        """Create prompt that constrains LLM to fill template safely"""
        
        prompt = f"""
You are generating a kubernetes scheduling policy function. You must ONLY fill in the logic between the comments.

CONSTRAINTS:
- Only use basic math operations (+, -, *, /, %, **, abs, min, max)
- Only use the provided variables: pod, node, cluster_state
- No imports, no function definitions, no loops
- Return a single numeric score
- Use if/else statements if needed
- Your generation should have nothing other than the code itself, do not output anything else. (Do not wrap in ```python)
- IMPORTANT: Every line of code MUST start with exactly 4 spaces for proper indentation
- Lines inside if/else blocks should start with 8 spaces, nested blocks with 12 spaces, etc.

Template to complete:
{cls.TEMPLATE}

Previous policies and their performance:
{cls._format_parent_policies(parent_policies)}

Performance feedback: {performance_feedback}

Generate ONLY the logic to replace {{llm_generated_logic}}, nothing else.
Remember: Each line must start with proper indentation (4 spaces minimum):
"""
        return prompt
    
    @classmethod
    def _format_parent_policies(cls, policies: list) -> str:
        """Format parent policies for prompt"""
        if not policies:
            return "No previous policies available."
        
        formatted = ""
        for i, (code, score) in enumerate(policies):
            formatted += f"\nPolicy v_{i+1} (score: {score:.3f}):\n{code}\n"
        return formatted
    
    @classmethod
    def fill_template(cls, llm_generated_logic: str) -> str:
        """Fill template with LLM-generated logic"""
        return cls.TEMPLATE.format(llm_generated_logic=llm_generated_logic.strip())


class LLMCodeGenerator:
    """Generate and validate scheduling policies using LLM"""
    
    def __init__(self, llm_client, safe_executor=None, model=None, max_tokens=400):
        self.llm_client = llm_client
        self.safe_executor = safe_executor or SafeExecutor()
        self.model = model or "gpt-3.5-turbo"
        self.max_tokens = max_tokens
    
    def generate_policy(self, parent_policies: list = None, 
                       performance_feedback: str = "") -> Optional[str]:
        """Generate new scheduling policy using LLM"""
        
        # Create constrained prompt
        prompt = PolicyTemplate.create_prompt_for_llm(
            parent_policies or [], 
            performance_feedback
        )
        
        try:
            # Get LLM response
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.max_tokens
            )
            
            llm_logic = response.choices[0].message.content.strip()
            
            # Fill template
            complete_code = PolicyTemplate.fill_template(llm_logic)
            
            print(complete_code)
            
            # Validate the generated code
            self.safe_executor.validate_code_content(complete_code)
            self.safe_executor.validate_code_structure(complete_code)
            
            return complete_code
            
        except Exception as e:
            print(f"Error generating policy: {e}")
            return None
    
    def test_policy_safely(self, code: str, test_pod, test_node) -> Optional[float]:
        """Test a policy function safely"""
        try:
            score = self.safe_executor.execute_policy_function(
                code, test_pod, test_node
            )
            return score
        except Exception as e:
            print(f"Error testing policy: {e}")
            return None