import gc
import math
import time
import random
import multiprocessing
import numpy as np
import torch


### universal tool functions
def get_noise_embeddings(embedding:np.ndarray, noise_ratio:float=0.0):
    if embedding is not None and noise_ratio > 0:
        emb_size = embedding.shape[-1]
        sample_list = list(range(emb_size))
        emb_num = embedding.shape[0]
        bs = 1024 if 1024 * 4 < emb_num else emb_num // 4
        bn = math.ceil(emb_num/bs)
        for i in range(bn):
            mask_id = random.sample(sample_list, int(emb_size * noise_ratio))
            embedding[i*bs:(i+1)*bs, mask_id] = 0
    return embedding

def div_list(ls, n):
    '''
    Split the input list into N parts
    
    Parameters
    ----------
    ls  : list
        the list to be splited
    n   : int
        the number that the list needs to be splited

    Returns
    -------
    out  :
        a list of length n, where each element is a split part of the original list ls
    '''
    if n <= 0:
        raise Exception(f"Error occured when split list : the paramter n = {n} <= 0.")
    if not isinstance(ls, list):
        raise Exception(f"Error occured when split list : the parameter ls is {type(ls)} , not list.")

    ls_len = len(ls)
    ls_return = []
    if n >= ls_len:
        ls_return = [[item] for item in ls]
    else:
        split_ls_len = ls_len // n
        split_ls_left = ls_len - split_ls_len * n
        actual_ls_lens = [split_ls_len] * n
        for i in range(split_ls_left):
            actual_ls_lens[i] += 1
        index = 0
        for i in range(n):
            ls_return.append(ls[index : index + actual_ls_lens[i]])
            index += actual_ls_lens[i]

    return ls_return


### torch tensor calculation
def tensor_broadcast(src:torch.Tensor, other:torch.Tensor, dim:int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src:torch.Tensor, index:torch.Tensor, dim:int=-1, out:torch.Tensor=None, dim_size:int=None) -> torch.Tensor:
    index = tensor_broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


### evaluate
def eval_alignment_by_ranks(ranks:list, hit_k:list=[1, 5, 10]):
    hits = [0] * len(hit_k)
    mrr = 0
    for r in ranks:
        mrr += 1 / (r + 1)
        for j in range(len(hit_k)):
            if r < hit_k[j]:
                hits[j] += 1
    total_num = len(ranks)
    mrr /= total_num
    hits = [hits[i]/total_num for i in range(hit_k)]
    return hits, mrr


### CSLS
def eval_alignment_by_embedding(embed1:np.ndarray, embed2:np.ndarray, top_k:list=[1, 5, 10], num_threads:int=16, csls:int=10, output:bool=True):
    sim_mat = np.matmul(embed1, embed2.T)
    return eval_alignment_by_sim_mat(sim_mat, top_k, num_threads, csls, output)

def eval_alignment_by_sim_mat(sim_mat:np.ndarray, top_k:list=[1, 5, 10], num_threads:int=16, csls:int=10, output:bool=True):
    st = time.time()
    sim_mat = sim_handler(sim_mat, csls, num_threads)
    
    ref_num = sim_mat.shape[0]
    tasks = div_list(np.array(range(ref_num)), num_threads)
    pool = multiprocessing.Pool(processes=num_threads)
    results = []
    for task in tasks:
        results.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k)))
    pool.close()
    pool.join()

    t_num = np.array([0 for _ in top_k])
    t_mean, t_mrr = 0, 0
    t_prec_set = set()
    for res in results:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = t_num / ref_num * 100
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        time_cost = time.time() - st
        print(f"accurate results: hits@{top_k} = {[round(a, 3) for a in acc]} , mr = {t_mean:.4f} , mrr = {t_mrr:.4f} , time_cost = {time_cost:.3f}s")
    del sim_mat
    gc.collect()
    return t_prec_set, acc, t_mrr


def sim_handler(sim_mat:np.ndarray, k, num_threads=16):
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, num_threads)
    csls2 = CSLS_sim(sim_mat.T, k, num_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat
    
def CSLS_sim(sim_mat:np.ndarray, k, num_threads=16) -> np.ndarray:
    tasks = div_list(np.array(range(sim_mat.shape[0])), num_threads)
    pool = multiprocessing.Pool(processes=num_threads)
    results = []
    for task in tasks:
        results.append(pool.apply_async(cal_csls_sim, (sim_mat[task, :], k)))
    pool.close()
    pool.join()

    sim_values = None
    for res in results:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    return sim_values

def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def cal_rank_by_sim_mat(task, sim, top_k):
    mean, mrr = 0, 0
    num = [0 for _ in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        rank = (-sim[i, :]).argsort()
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j, k in enumerate(top_k):
            if rank_index < k:
                num[j] += 1
    return mean, mrr, num, prec_set


### candidate entities
def generate_candidates_by_sim_mat(sim_mat:np.ndarray, ent1, ent2, cand_num, csls=10, num_thread=16):
    '''
    Output: {ent_id_1:{'ground_rank': rank, 'candidates': candidates}}
        - ground_rank : the rank of entity2 in entity_list2 which matches ent_id_1, according to reference pairs
        - candidates  : top K entities which matches ent_id_1, according to sim_mat = dot(embed1, embed2)
    '''
    ent_idx_1 = {e:i for i, e in enumerate(ent1)}
    ent_frags = div_list(np.array(ent1), num_thread)
    sim_mat = sim_handler(sim_mat, csls, num_thread)
    ref_num = sim_mat.shape[0]
    tasks = div_list(np.array(range(ref_num)), num_thread)

    pool = multiprocessing.Pool(processes=len(tasks))
    results = []
    for i, task in enumerate(tasks):
        results.append(pool.apply_async(find_candidates_by_sim_mat, args=(ent_frags[i], ent_idx_1, sim_mat[task, :], np.array(ent2), cand_num)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic = {**dic, **res.get()}
    del results
    gc.collect()
    return dic

def find_candidates_by_sim_mat(frags, ent_idx, sim, entity_list2, k):
    dic = {}
    for i in range(sim.shape[0]):
        ref = ent_idx[frags[i]]
        rank = (-sim[i, :]).argsort()
        rank_index = np.where(rank == ref)[0][0]
        cand_index = np.argpartition(-sim[i, :], k)[:k]
        candidates = entity_list2[cand_index].tolist()
        cand_sims = [float(s) for s in sim[i, cand_index]]
        min_s, max_s = min(cand_sims), max(cand_sims)
        cand_sims = [(s - min_s) / (max_s - min_s) for s in cand_sims]
        
        sorted_cand = sorted([(candidates[j], cand_sims[j]) for j in range(k)], key=lambda x: x[1], reverse=True)
        candidates = [c_idx for c_idx, _ in sorted_cand]
        cand_sims  = [c_sim for _, c_sim in sorted_cand]

        dic[int(frags[i])] = {'ground_rank':int(rank_index), 'candidates':candidates, 'cand_sims':cand_sims}
    return dic