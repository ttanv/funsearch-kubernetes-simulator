from dataclasses import dataclass

"""The 'bins' of the problem"""
@dataclass
class GPU:
    """Represents single GPU inside a node"""
    gpu_id: int
    memory_mib_left: int
    memory_mib_total: int
    
@dataclass
class Node:
    """Represents a node that may contain multiple GPUs"""
    node_id: int
    cpu_milli_left: int
    cpu_milli_total: int
    memory_mib_left: int
    memory_mib_total: int
    gpus: list[GPU]
    
@dataclass
class Pod:
    """Represents a pod request"""
    pod_id: int
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    gpu_spec: str
    creation_time: int
    deletion_time: int
    