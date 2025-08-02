from dataclasses import dataclass

"""The 'bins' of the problem"""
@dataclass
class GPU:
    """Represents single GPU inside a node"""
    memory_mib_left: int
    memory_mib_total: int
    
@dataclass
class Node:
    """Represents a node that may contain multiple GPUs"""
    node_id: str
    cpu_milli_left: int
    cpu_milli_total: int
    memory_mib_left: int
    memory_mib_total: int
    gpu_count: int
    gpus: list[GPU]
    
@dataclass
class Cluster:
    """Represents the full cluster"""
    nodes_dict: dict[str, Node]    

"""The items of the problem"""    
@dataclass
class Pod:
    """Represents a pod request"""
    pod_id: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    gpu_spec: str
    creation_time: int
    deletion_time: int
    