import os
import json
import numpy as np
import scipy.sparse as sp



def normalize_adj(adj) -> sp.csr_matrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T
    return norm_adj


class KGDataLoader:
    def __init__(self, data_dir:str):
        '''
        KGDataLoader Parameters
        ------------------------
        data_dir    : str
            the directory of datasets
            triples file, pairs file and so on should be under this directory
        image_paths : dict
            the dictionary that maps ent_id to image paths
            in format : {
                ent_id_0 : [path00, path01, ...],
                ent_id_1 : [path10, path11, ...],
                ......
            }
        '''
        ### flag of which info will be used
        self.has_time = True

        ### basic data
        self.data_dir = data_dir
        if self.data_dir is None:
            raise Exception("Directory of data should not be None !")
        if not isinstance(self.data_dir, str):
            raise Exception(f"Directory of data should be a string, not {type(self.data_dir)}")
        self.image_paths = {}
        self.load_image_paths_by_paths(os.path.join(self.data_dir, "image_path"))
        
                
        ###### knowledge triples
        self.load_triples_by_paths([os.path.join(self.data_dir, "triples_1"), os.path.join(self.data_dir, "triples_2")])
        ###### id2name and name2id dictionary
        self.load_entity_names_by_paths([os.path.join(self.data_dir, "ent_ids_1"), os.path.join(self.data_dir, "ent_ids_2")])
        self.load_relation_names_by_paths([os.path.join(self.data_dir, "rel_ids_1"), os.path.join(self.data_dir, "rel_ids_2")])
        if self.has_time:
            self.load_time_names_by_paths([os.path.join(self.data_dir, "time_id")])
        ###### entity pairs
        self.load_sup_ref_pairs_by_paths([os.path.join(self.data_dir, "sup_pairs"), os.path.join(self.data_dir, "ref_pairs")])
        ###### all entities in KG
        self.entity_pair_dict = self.process_entity_pairs()
        self.entities1 = self.process_entities_in_KG(KG_id=1)
        self.entities2 = self.process_entities_in_KG(KG_id=2)
        self.entities = sorted(list(set(self.entities1 + self.entities2)))
        ###### process KG to matrix
        self.adj_matrix, self.adj_features, self.rel_features, self.r_index, self.r_val = self.process_KGs_to_matrix()
    

    ### API
    def ent_has_name(self, ent_id:int):
        return ent_id in self.ent_id2name
    def rel_has_name(self, rel_id:int):
        return rel_id in self.rel_id2name
    def get_num_of_entity(self):
        return int(max(self.ent_id2name.keys())) + 1
    def get_num_of_relation(self):
        return int(max(self.rel_id2name.keys())) + 1
    def get_name_by_eid(self, ent_id:int):
        return self.ent_id2name[ent_id] if ent_id in self.ent_id2name else None
    def get_name_by_rid(self, rel_id:int):
        return self.rel_id2name[rel_id] if rel_id in self.rel_id2name else None
    def get_name_by_tid(self, time_id:int):
        return self.time_id2name[time_id] if time_id in self.time_id2name else None
    def get_eid_by_name(self, ent_name:str):
        return self.ent_name2id[ent_name] if ent_name in self.ent_name2id else None
    def get_rid_by_name(self, rel_name:str):
        return self.rel_name2id[rel_name] if rel_name in self.rel_name2id else None
    def get_pair_ent(self, ent_id:int):
        return self.entity_pair_dict[ent_id] if ent_id in self.entity_pair_dict else None
    
    def get_all_entities(self):
        return self.entities
    def get_all_triples(self):
        return self.triples1 + self.triples2
    def get_all_pairs(self):
        return np.concatenate([self.sup_pairs, self.ref_pairs])
    def get_KG_entities(self, KG_id:int):
        if KG_id not in [1, 2]:
            raise Exception("Error : KG's id not in [1, 2]")
        if KG_id == 1:
            return self.entities1
        else:
            return self.entities2
    def get_KG_triples(self, KG_id:int):
        if KG_id not in [1, 2]:
            raise Exception("Error : KG's id not in [1, 2]")
        if KG_id == 1:
            return self.triples1
        else:
            return self.triples2
    
    ### function for process datas
    def process_entity_pairs(self):
        entity_pairs:dict[int, int] = {}
        for e1, e2 in self.sup_pairs + self.ref_pairs:
            entity_pairs[e1] = e2
            entity_pairs[e2] = e1
        return entity_pairs

    def process_entities_in_KG(self, KG_id:int):
        id_path = os.path.join(self.data_dir, f"ent_ids_{KG_id}")
        ent_id2name = {}
        if os.path.exists(id_path):
            ent_id2name, _ = self.load_id_dict(id_path)
        ent_ids = set(ent_id2name.keys())

        if len(ent_ids) == 0:
            triples = self.triples1 if KG_id == 1 else self.triples2
            for triple in triples:
                h, _, t = triple[:3]
                ent_ids.add(h)
                ent_ids.add(t)

        ent_ids = sorted(list(ent_ids))
        if KG_id == 1:
            self.entities1 = ent_ids
        elif KG_id == 2:
            self.entities2 = ent_ids
        return ent_ids

    def process_KGs_to_matrix(self):
        ent_size = self.get_num_of_entity()
        rel_size = self.get_num_of_relation()
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        radj = []
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))

        for i in range(ent_size):
            adj_features[i, i] = 1
        for triple in self.get_all_triples():
            h, r, t = triple[:3]
            adj_matrix[h, t] = 1; adj_matrix[t, h] = 1
            adj_features[h, t] = 1; adj_features[t, h] = 1
            radj.append([h, t, r]); radj.append([t, h, r+rel_size])
            rel_out[h][r] += 1; rel_in[t][r] += 1

        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x:x[0]*10e10+x[1]*10e5):
            if f"{h} {t}" not in s:
                count += 1
                s.add(f"{h} {t}")
            d[count] += 1
            r_index.append([count, r])
            r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(sp.lil_matrix(rel_features))

        r_index, r_val = np.array(r_index), np.array(r_val)
        self.adj_matrix = adj_matrix
        self.adj_features = adj_features
        self.rel_features = rel_features
        self.r_index = r_index
        self.r_val = r_val
        return adj_matrix, adj_features, rel_features, r_index, r_val
        

    ### function for loading datas
    def load_id_dict(self, path:str, processed:bool=False):
        id2name:dict[int, str] = {}
        name2id:dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                items = line.strip().split("\t")
                if items[0].isdigit():
                    idx, item = int(items[0]), str(items[1])
                else:
                    idx, item = int(items[1]), str(items[0])
                if processed:
                    item = item.split("/")[-1].replace(u"\xa0", "")
                id2name[idx] = item
                name2id[item] = idx
        return id2name, name2id

    def load_name_dict(self, files:list, processed=False):
        ### load name--id dict
        id2name, name2id = {}, {}
        for file_path in files:
            if os.path.exists(file_path):
                single_id2name, single_name2id = self.load_id_dict(file_path, processed=processed)
                id2name, name2id = {**id2name, **single_id2name}, {**name2id, **single_name2id}
        return id2name, name2id

    def load_time_dict(self, files:list=["time_id"]):
        time_id2name = {}
        file_exists = True
        for p in files:
            if not os.path.exists(p):
                file_exists = False
                break
        if file_exists:
            for p in files:
                time_dict, _ = self.load_id_dict(p)
                time_id2name = {**time_id2name, **time_dict}
        else:
            time_size = 0
            for i in range(2):
                file_path = os.path.join(self.data_dir, f"triples_{i+1}")
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as fr:
                        for line in fr.readlines():
                            items = [int(item) for item in line.strip().split("\t")]
                            if len(items) <= 3 or len(items) > 5:
                                raise Exception(f"Error occured when loading time dict : length of triples in {file_path} is {len(items)} , which means there is no time infomation in these triples.")
                            else:
                                if len(items) == 4:
                                    time_size = max(time_size, items[3])
                                elif len(items) == 5:
                                    time_size = max(time_size, items[3], items[4])
            for tid in range(time_size):
                time_id2name[tid] = str(tid)
        for idx, time in time_id2name.items():
            if time == "" or time.startswith("-") :
                time_id2name[idx] = "~"

        self.time_id2name = time_id2name
        return time_id2name

    def load_entity_pairs(self, path:str):
        pairs = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    e1, e2 = [int(e) for e in line.strip().split("\t")]
                    pairs.append((e1, e2))
        return np.array(pairs)

    def load_triples(self, path:str):
        triples = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    items = [int(item) for item in line.strip().split("\t")]
                    h, r, t = items[:3]
                    if len(items) < 3 or len(items) > 5:
                        self.has_time = False
                        raise Exception(f"Error occured when loading triples : length of triples in {path} is {len(items)}, not in [3, 4, 5].")
                    else:
                        if len(items) == 3:
                            triple = (h, r, t)
                            self.has_time = False
                        elif len(items) == 4:
                            tau = items[3]
                            triple = (h, r, t, tau, tau)
                        elif len(items) == 5:
                            ts, te = items[3:]
                            triple = (h, r, t, ts, te)
                    triples.append(triple)
        return triples

    ### API for loading datas
    def load_triples_by_paths(self, paths:list):
        '''
        Paramters
        ----------
        paths   : list[str]
            [ path_of_triples_in_KG_1 , path_of_triples_in_KG_2 ]

        Return
        ----------
        triples_in_KG_1   : list
            triples (h, r, t) or (h, r, t, ts, te) in KG 1
        triples_in_KG_2   : list
            triples in KG 2
        '''
        if len(paths) != 2:
            raise Exception(f"Error occured when loading triples : input {len(paths)} > 2 KGs, check the input param : triple paths")
        self.triples1 = self.load_triples(paths[0])
        self.triples2 = self.load_triples(paths[1])
        return self.triples1, self.triples2
    
    def load_entity_names_by_paths(self, paths:list):
        self.ent_id2origin_name, _ = self.load_name_dict(paths, processed=False)
        self.ent_id2name, self.ent_name2id = self.load_name_dict(paths, processed=True)
        return self.ent_id2name, self.ent_name2id

    def load_relation_names_by_paths(self, paths:list):
        self.rel_id2name, self.rel_name2id = self.load_name_dict(paths, processed=True)
        return self.rel_id2name, self.rel_name2id

    def load_time_names_by_paths(self, paths:list):
        self.time_id2name = self.load_time_dict(paths)
        return self.time_id2name

    def load_sup_ref_pairs_by_paths(self, paths:list, train_ratio:float=0.0):
        '''
        Paramters
        ----------
        paths   : list[str]
            [ path_of_sup_pairs/train , path_of_ref_pairs/test ] or [ path_or_all_pairs ]
        train_ratio : float
            if the paths param is in format [ path_of_all_pairs ] , set train_ratio > 0 to split the pairs into sup/train pairs and ref/test pairs.

        Return
        ----------
        sup_pairs   : np.ndarray
            train/supervised entity pairs
        ref_pairs   : np.ndarray
            test/reference entity pairs
        '''
        if len(paths) > 2:
            raise Exception(f"Error occured when loading pairs : input {len(paths)} > 2 files.")
        elif len(paths) == 2:
            ### [ path_of_sup_pairs/train , path_of_ref_pairs/test ]
            self.sup_pairs = self.load_entity_pairs(paths[0])
            self.ref_pairs = self.load_entity_pairs(paths[1])
        else:
            ### [ path_of_all_pairs ]
            if train_ratio > 0:
                all_pairs = self.load_entity_pairs(paths[0])
                np.random.shuffle(all_pairs)
                train_split = int(len(all_pairs) * train_ratio)
                self.sup_pairs = all_pairs[:train_split]
                self.ref_pairs = all_pairs[train_split:]
            else:
                raise Exception(f"Error occured when loading entity pairs : only input 1 file , but train_ratio == 0.0")
        return self.sup_pairs, self.ref_pairs
        
    def load_image_paths_by_paths(self, path:str=None):
        '''
        Paramters
        ---------
        path  : str
            path of file which stores the image paths in json format
        
        Return
        ---------
        image_paths  : dict
            image_paths in format : {ENT_ID : {"root" : ROOT, "file" : [FILE0, FILE1, ...] } , ... }
            the final image_path of ENT_ID will be "DATA_DIR / ROOT / FILE0"
        '''
        if path is not None:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fr:
                    self.image_paths = json.load(fr)

        if not isinstance(list(self.image_paths.keys())[0], int):
            new_image_paths = {}
            for eid, value in self.image_paths.items():
                new_image_paths[int(eid)] = value
            self.image_paths = new_image_paths