import pickle
import numpy as np
import scipy.sparse as sp



def normalize_embed_dict(ent_num:int, embed_dict:dict) -> np.ndarray:
    feat_np = np.array(list(embed_dict.values()))
    mean, std = np.mean(feat_np, axis=0), np.std(feat_np, axis=0)
    feat_embed = np.array([embed_dict[i] if i in embed_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(ent_num)])
    embeddings = feat_embed / np.linalg.norm(feat_embed, axis=-1, keepdims=True)
    return embeddings


def load_embeddings(ent_num:int, data_path:str):
    if data_path.endswith(".txt"):
        embeddings = np.loadtxt(data_path)
    elif data_path.endswith(".npy"):
        embeddings = np.load(data_path)
    else:
        embeddings = pickle.load(open(data_path, "rb"))
        if isinstance(embeddings, dict):
            embeddings = normalize_embed_dict(embeddings)
        else:
            embeddings = np.array(embeddings)
    if embeddings.shape[0] != ent_num:
        raise Exception(f"Error occured when load embeddings : embeddings are in shape {embeddings.shape} , where the length of dim 0 is not equal to entity number.")
    return embeddings


def load_embeddings_from_paths(ent_num:int, paths) -> np.ndarray:
    ### path check
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, list):
        paths = paths
    else:
        raise Exception(f"Error occured when loading embeddings : please input a list of embedding paths")
    if len(paths) == 0:
        raise Exception(f"Error occured when loading embeddings : the path list is empty !")
    ### load and merge embeddings
    embedding_list = [load_embeddings(ent_num, path).tolist() for path in paths]
    embeddings = embedding_list[0]
    for embed in embedding_list[1:]:
        embeddings = embeddings + embed
    embeddings = np.array(embeddings)
    return embeddings