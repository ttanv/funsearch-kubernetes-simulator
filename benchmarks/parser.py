import csv
import json
from typing import Dict, List, Optional
from pathlib import Path

from simulator.entities import Node, GPU, Cluster, Pod


class TraceParser:
    """Parser for OpenB dataset traces to populate simulation entities"""
    
    def __init__(self, traces_dir: str = "benchmarks/traces"):
        self.traces_dir = Path(traces_dir)
        self.csv_dir = self.traces_dir / "csv"
        self.gpu_mem_mapping = self._load_gpu_memory_mapping()
    
    def _load_gpu_memory_mapping(self) -> Dict[str, int]:
        """Load GPU memory mapping from JSON file"""
        mapping_file = self.traces_dir / "gpu_mem_mapping.json"
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    def parse_nodes(self, node_file: str = "openb_node_list_all_node.csv") -> Dict[str, Node]:
        """Parse node data from CSV file and create Node entities"""
        nodes = {}
        node_path = self.csv_dir / node_file
        
        with open(node_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row['sn']
                cpu_total = int(row['cpu_milli'])
                memory_total = int(row['memory_mib'])
                gpu_count = int(row['gpu'])
                gpu_model = row['model']
                
                # Create GPU entities for this node
                gpus = []
                if gpu_count > 0 and gpu_model in self.gpu_mem_mapping:
                    gpu_memory = self.gpu_mem_mapping[gpu_model]
                    for _ in range(gpu_count):
                        gpu = GPU(
                            memory_mib_left=gpu_memory,
                            memory_mib_total=gpu_memory,
                            gpu_milli_left=1000,
                            gpu_milli_total=1000
                        )
                        gpus.append(gpu)
                
                node = Node(
                    node_id=node_id,
                    cpu_milli_left=cpu_total,
                    cpu_milli_total=cpu_total,
                    memory_mib_left=memory_total,
                    memory_mib_total=memory_total,
                    gpu_left=gpu_count,
                    gpus=gpus,
                )
                nodes[node_id] = node
        
        return nodes
    
    def parse_cluster(self, node_file: str = "openb_node_list_gpu_node.csv") -> Cluster:
        """Parse cluster data from node file"""
        nodes = self.parse_nodes(node_file)
        return Cluster(nodes_dict=nodes)
    
    def parse_pods(self, pod_file: str = "openb_pod_list_default.csv") -> List[Pod]:
        """Parse pod data from CSV file and create Pod entities"""
        pods = []
        pod_path = self.csv_dir / pod_file
        
        with open(pod_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Use pod name as ID
                pod_id = row['name']
                
                cpu_milli = int(row['cpu_milli'])
                memory_mib = int(row['memory_mib'])
                num_gpu = int(row['num_gpu'])
                gpu_milli = int(row['gpu_milli']) if row['gpu_milli'] else 0
                gpu_spec = row['gpu_spec'] if row['gpu_spec'] else ""
                creation_time = int(row['creation_time'])
                deletion_time = int(row['deletion_time'])
                
                pod = Pod(
                    pod_id=pod_id,
                    cpu_milli=cpu_milli,
                    memory_mib=memory_mib,
                    num_gpu=num_gpu,
                    gpu_milli=gpu_milli,
                    gpu_spec=gpu_spec,
                    creation_time=creation_time,
                    duration_time=deletion_time - creation_time,
                    assigned_node="",
                    assigned_gpus=[]
                )
                pods.append(pod)
        
        return pods
    
    def get_available_node_files(self) -> List[str]:
        """Get list of available node CSV files"""
        node_files = []
        for file in self.csv_dir.glob("openb_node_list_*.csv"):
            node_files.append(file.name)
        return sorted(node_files)
    
    def get_available_pod_files(self) -> List[str]:
        """Get list of available pod CSV files"""
        pod_files = []
        for file in self.csv_dir.glob("openb_pod_list_*.csv"):
            pod_files.append(file.name)
        return sorted(pod_files)
    
    def parse_workload(self, node_file: str = "openb_node_list_gpu_node.csv", 
                      pod_file: str = "openb_pod_list_default.csv") -> tuple[Cluster, List[Pod]]:
        """Parse both cluster and pods from trace files"""
        cluster = self.parse_cluster(node_file)
        pods = self.parse_pods(pod_file)
        return cluster, pods


def main():
    """Example usage of the parser"""
    parser = TraceParser()
    
    print("Available node files:", parser.get_available_node_files())
    print("Available pod files:", parser.get_available_pod_files())
    
    # Parse default workload
    cluster, pods = parser.parse_workload()
    
    print(f"\nParsed cluster with {len(cluster.nodes_dict)} nodes")
    print(f"Parsed {len(pods)} pods")
    
    # Print sample node info
    if cluster.nodes_dict:
        sample_node = next(iter(cluster.nodes_dict.values()))
        print(f"\nSample node: {sample_node.node_id}")
        print(f"  CPU: {sample_node.cpu_milli_total} milli")
        print(f"  Memory: {sample_node.memory_mib_total} MiB")
        print(f"  GPUs: {sample_node.gpu_left}")
        if sample_node.gpus:
            print(f"  GPU Memory: {sample_node.gpus[0].memory_mib_total} MiB each")
    
    # Print sample pod info
    if pods:
        sample_pod = pods[0]
        print(f"\nSample pod: {sample_pod.pod_id}")
        print(f"  CPU: {sample_pod.cpu_milli} milli")
        print(f"  Memory: {sample_pod.memory_mib} MiB") 
        print(f"  GPUs: {sample_pod.num_gpu}")
        print(f"  Creation time: {sample_pod.creation_time}")


if __name__ == "__main__":
    main()