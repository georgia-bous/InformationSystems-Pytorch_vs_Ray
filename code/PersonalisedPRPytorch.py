from torch_ppr import personalized_page_rank
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pyarrow as pa
import time

def setup_distributed(rank, world_size):
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = '192.168.0.2'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

'''
#READ FROM LOCAL
class NodeDataset(Dataset):
    def __init__(self, file_path):
        self.nodes=list(range(4))
        self.edge_index = self.load_graph(file_path)

    def load_graph(self, file_path):
        nodes = set()
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                node1, node2 = map(int, line.split())
                edges.append([node1, node2])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

'''
#READ FROM HDFS
class NodeDataset(Dataset):
    def __init__(self, file_path, hdfs_host, hdfs_port):
        self.nodes = list(range(4))  #only first 4 nodes for faster execution
        self.edge_index = self.load_graph(file_path, hdfs_host, hdfs_port)

    def load_graph(self, file_path, hdfs_host, hdfs_port):
        hdfs = pa.hdfs.connect(hdfs_host, hdfs_port)
        edges = []
        buffer = ""
        with hdfs.open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024*1024*50)  # Read 50MB at a time
                if not chunk:
                    break
                chunk_str = buffer + chunk.decode()
                lines = chunk_str.split('\n')
                buffer = lines.pop()  # Save incomplete line for the next chunk
                for line in lines:
                    if not line.strip():
                        continue
                    node1, node2 = map(int, line.split())
                    edges.append([node1, node2])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]


def distributed_ppr(rank, world_size, node_dataset, dataloader):
    device="cpu"
    all_ppr_scores = []
    for batch in dataloader:
        batch=batch.to(device)
        batch_scores = personalized_page_rank(edge_index=node_dataset.edge_index,indices= batch)
        all_ppr_scores.extend(batch_scores)

    all_ppr_scores = torch.stack(all_ppr_scores)
    gathered_scores = [torch.zeros_like(all_ppr_scores) for _ in range(world_size)]
    dist.all_gather(gathered_scores, all_ppr_scores)
    # Combine gathered results
    if rank == 0:
        combined_results = torch.cat(gathered_scores, dim=0)
        return combined_results
    else:
        return None  # Non-root processes return None


def main():
    start_time = time.time()
    rank = 0  
    world_size = 2  
    hdfs_host = "okeanos-master"
    hdfs_port = 54310
    shuffle = False
    batch_size = 2

    setup_distributed(rank, world_size)
    # Load data on each node
    #file_path = "twitter-2010-subset.txt"  
    #node_dataset=NodeDataset(file_path)
    file_path = "hdfs://okeanos-master:54310/twitter-2010-subset.txt"
    file_path = "hdfs://okeanos-master:54310/twitter-2010-sub7.txt"
    node_dataset = NodeDataset(file_path, hdfs_host, hdfs_port)
    sampler = DistributedSampler(node_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    dataloader = DataLoader(node_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    
    print("Length of DataLoader in process {}: {}".format(rank, len(dataloader)))
    load=time.time()
    print("Load:", load-start_time, "seconds.")
    result=distributed_ppr(rank, world_size, node_dataset, dataloader)
    if rank==0:
        print(result[:10])
        print(result.shape)

    # Finalize distributed environment
    dist.destroy_process_group()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()