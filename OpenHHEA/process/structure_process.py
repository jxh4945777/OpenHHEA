import os
import gc
import numpy as np
import networkx as nx
import gensim
from gensim.models import Word2Vec

from OpenHHEA.configs.types import ProcessStruct
from OpenHHEA.data import KGDataLoader
from .utils_dataprocess import normalize_embed_dict
from .longterm.node2vec import Graph



class StructProcessor:
    def __init__(self) -> None:
        pass

    def struct_process(self, loader:KGDataLoader) -> np.ndarray:
        pass


class FualignStructProcessor(StructProcessor):
    def __init__(self, 
        weighted:bool=False,
        directed:bool=False,
        walk_length:int=80,
        num_walks:int=10,
        p:float=1e-100,
        q:float=1.0,
        dim:int=64,
        window_size:int=10,
        workers:int=12,
        iter:int=5
    ) -> None:
        super().__init__()
        self.weighted = weighted
        self.directed = directed

        self.walk_length = walk_length
        self.num_walks = num_walks

        self.p = p
        self.q = q
        self.dim = dim
        self.window_size = window_size
        self.workers = workers
        self.iter = iter

        self.tmp_dir = "temp_data"
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def get_deepwalk_data(self, 
        loader:KGDataLoader, 
        deepwalk_file:str="deepwalk.data"
    ):
        # process same entity
        ent_num = loader.get_num_of_entity()
        node2same = {eid:eid for eid in loader.get_all_entities()}
        for i, (e1, e2) in enumerate(loader.sup_pairs):
            node2same[e1] = ent_num + i
            node2same[e2] = ent_num + i
        # process same relation
        rel_num = loader.get_num_of_relation()
        same_rel = {}
        for id, name in loader.rel_id2name.items():
            if name in same_rel:
                same_rel[name].append(id)
            else:
                same_rel[name] = [id]
        rel2same = {}
        for i, ids in same_rel.values():
            if len(ids) > 1:
                for id in ids:
                    rel2same[id] = rel_num + i
        # deepwalk data
        deepwalk_data = []
        node2rel, rel2index = {}, {}
        start_index, rel_count_index = max(node2same.values()) + 1, 0
        for triple in loader.get_all_triples():
            h, r, t = node2same[triple[0]], rel2same[triple[1]], node2same[triple[2]]
            deepwalk_data.append((h, t))
            if r not in rel2index:
                rel2index[r] = start_index + rel_count_index
                rel_count_index += 1
            r = rel2index[r]
            node2rel[f"{h}+{t}"] = r
            node2rel[f"{t}+{h}"] = r
        # save data
        tmp_deep_data_path = os.path.join(self.tmp_dir, deepwalk_file)
        with open(tmp_deep_data_path, "w") as fw:
            for h, t in deepwalk_data:
                fw.write(f"{h} {t}\n")
        return tmp_deep_data_path, node2same, node2rel

    def struct_process(self, loader: KGDataLoader, remove_tmp:bool=False) -> np.ndarray:
        tmp_deep_data_path, node2same, node2rel = self.get_deepwalk_data(loader, deepwalk_file="deepwalk.data")
        ### longterm
        if self.weighted:
            G = nx.read_edgelist(tmp_deep_data_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(tmp_deep_data_path, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]["weight"] = 1
        if not self.directed:
            G = G.to_undirected()
        print("read graph finished...")
        G = Graph(G, is_directed=self.directed, p=self.p, q=self.q)
        print("node2vec finished...")
        G.preprocess_transition_probs()
        print("preprocess transition probs finished...")
        walks = G.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length)
        print("simulate walks finished...")
        temp = []
        for line in walks:
            new_line = [line[0]]
            for i in range(len(line) - 1):
                h, t = str(line[i]), str(line[i+1])
                r = str(node2rel[f"{h}+{t}"])
                new_line += [r, t]
            temp.append(new_line)
        walks = [list(map(str, walk)) for walk in temp]
        model = Word2Vec(walks, vector_size=self.dim, window=self.window_size, min_count=0, sg=1, workers=self.workers, epochs=self.iter)
        tmp_longterm_vec_path = os.path.join(self.tmp_dir, "longterm.vec")
        model.wv.save_word2vec_format(tmp_longterm_vec_path)
        ### get deep emb
        deep_wv = gensim.models.KeyedVectors.load_word2vec_format(tmp_longterm_vec_path)
        struct_embed = {}
        for id in loader.get_all_entities():
            new_id = node2same[id]
            if new_id in deep_wv:
                struct_embed[id] = deep_wv[new_id]
        if remove_tmp:
            if os.path.exists(tmp_deep_data_path):
                os.remove(tmp_deep_data_path)
            if os.path.exists(tmp_longterm_vec_path):
                os.remove(tmp_longterm_vec_path)
        ### normalize embeddings
        embeddings = normalize_embed_dict(loader.get_num_of_entity(), struct_embed)
        return embeddings


class GCNStructProcessor(StructProcessor):
    def __init__(self, 

    ) -> None:
        super().__init__()
        


# unified inference function
processor_dict = {
    ProcessStruct.FUALIGN : FualignStructProcessor
}
def get_struct_processor(method_type, **kwargs) -> StructProcessor:
    if method_type not in processor_dict:
        raise Exception(f"Error occured when process name : there is no struct processor using {method_type}")
    return processor_dict[method_type](**kwargs)

def struct_embedding_process(method_type, dataloader:KGDataLoader, **kwargs) -> np.ndarray:
    print("-------------------- Process Structure Information --------------------")
    if method_type not in ProcessStruct:
        raise Exception(f"Error occured when process structure : there is no structure process method {method_type}")
    processor = get_struct_processor(method_type, **kwargs)
    embeddings = processor.struct_process(dataloader)
    gc.collect()
    return embeddings