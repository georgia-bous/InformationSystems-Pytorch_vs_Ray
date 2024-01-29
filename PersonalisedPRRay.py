import ray
import torch
from torch.utils.data import Dataset, DataLoader
from torch_ppr import personalized_page_rank
import pyarrow as pa
import time

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
    def __init__(self, file_path, hdfs_host='okeanos-master', hdfs_port=54310):
        self.nodes = list(range(4))
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

@ray.remote
def distributed_ppr(edge_index, nodes):
    device = "cpu"
    all_ppr_scores = []
    for node in nodes:
        node = torch.tensor([node]).to(device)
        batch_scores = personalized_page_rank(edge_index=edge_index, indices=node)
        all_ppr_scores.extend(batch_scores)
    return torch.stack(all_ppr_scores)

def main():
    start_time = time.perf_counter()
    ray.init()  # Initialize Ray
    shuffle=False
    batch_size=2

    #file_path = "twitter-2010-subset.txt"
    file_path = "/twitter-2010-sub7.txt"
    node_dataset = NodeDataset(file_path)
    dataloader = DataLoader(node_dataset, batch_size=batch_size, shuffle=shuffle)

    # Distribute computations
    #non blocking call of distributed_ppr for each batch, gather the results to a list
    load=time.perf_counter()
    print(f"Load: {load-start_time} seconds")
    futures = [distributed_ppr.remote(node_dataset.edge_index, batch) for batch in dataloader]
    #block until execution of all tasks is finished
    results = ray.get(futures)
    # Combine results
    combined_results = torch.cat(results, dim=0)
    print(combined_results[:10])
    print(combined_results.shape)

    ray.shutdown()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")




if __name__ == "__main__":
    main()