import os
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from OpenHHEA.configs.model_configs import XGEAConfig
from OpenHHEA.data import KGDataLoader
from OpenHHEA.process import KGDataProcessor
from OpenHHEA.evaluate import eval_alignment_by_embedding
from OpenHHEA.reasoning.utils import get_noise_embeddings, scatter_sum
from OpenHHEA.reasoning.results import XGEAResult


### Main Pipeline of XGEA
class PipelineXGEA:
    def __init__(self, 
        config:XGEAConfig,
        dataloader:KGDataLoader,
        dataprocessor:KGDataProcessor,
        model_path:str=None
    ):
        self.config = config
        self.loader = dataloader
        self.processor = dataprocessor
        self.entity_embeddings = None

        train_pairs, adj_matrix, ent_matrix, rel_matrix = self.__xgea_data_process()
        self.train_pairs = train_pairs

        node_size = self.loader.get_num_of_entity()
        rel_size = self.loader.get_num_of_relation()
        triple_size = len(adj_matrix)
        if self.config.use_literal:
            features = self.processor.name_embeddings
        else:
            features = self.processor.image_embeddings

        self.model = XGEA(
            node_hidden=self.config.node_hidden_size,
            rel_hidden=self.config.rel_hidden_size,
            node_size=node_size,
            rel_size=rel_size,
            triple_size=triple_size,
            adj_matrix=adj_matrix,
            ent_matrix=ent_matrix,
            rel_matrix=rel_matrix,
            features=get_noise_embeddings(features, self.config.embed_noise_ratio),
            dropout_rate=self.config.dropout_rate,
            depth=self.config.depth,
            device=self.config.device
        ).to(self.config.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))
            self.update_entity_embeddings()

        self.best_model_path = os.path.join(self.config.model_save_dir, f"best_xgea_model_noise_{self.config.embed_noise_ratio}.pth")


    def run(self, save_result_path:str=None):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")

        st = time.time()
        losses, t_prec, accs, t_mrrs, best_acc, best_mrr = self.train(
            train_alignment=self.train_pairs,
            dev_alignments=self.loader.ref_pairs,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.lr,
            weight_decay=self.config.weight_decay,
            gamma=self.config.gamma,
            hit_k=self.config.hit_k
        )

        self.update_entity_embeddings()

        result = XGEAResult(
            config=self.config,
            hits=best_acc,
            mrr=best_mrr,
            time_cost=time.time() - st
        )
        if save_result_path is not None:
            save_dir = os.path.split(save_result_path)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_result_path, "a+", encoding="utf-8") as fw:
                fw.write("\n" + str(result) + "\n")

        return result

    
    def train(self, train_alignment:np.ndarray, dev_alignments:np.ndarray, epochs=50, batch_size=1024, learning_rate=0.01, weight_decay=0.0005, gamma=1.0, hit_k=[1, 5, 10]):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        rest_set_1 = [e1 for e1, _ in dev_alignments]
        rest_set_2 = [e2 for _, e2 in dev_alignments]
        np.random.shuffle(rest_set_1)
        np.random.shuffle(rest_set_2)

        losses, t_prec, accs, t_mrrs = [], [], [], []
        best_acc = [0] * len(hit_k)
        best_mrr = 0
        train_pairs = train_alignment
        for turn in range(self.config.turn):
            print(f"\n### Iteration {turn + 1}")
            for i in tqdm(range(epochs), desc=f"Iteration {turn + 1}"):
                self.model.train()
                optimizer.zero_grad()
                for pairs in self.__get_train_batches(train_pairs, batch_size):
                    features_join = self.model()
                    loss = self.__get_align_loss(pairs, features_join, gamma)
                    losses.append(loss.item())

                    loss.backward()
                    optimizer.step()
                ### evaluate
                if (i + 1) % self.config.eval_step == 0:
                    t_prec_set, acc, t_mrr = self.evaluate(dev_alignments, hit_k, self.config.eval_threads)
                    for i in range(len(hit_k)):
                        if best_acc[i] < acc[i]:
                            best_acc[i] = acc[i]
                    if best_mrr < t_mrr:
                        best_mrr = t_mrr
                        torch.save(self.model.state_dict(), self.best_model_path)
                    best_acc = [round(n, 3) for n in acc]
                    print(f"// best results : hits@{hit_k} = {best_acc} , mrr = {best_mrr:.4f} //")
                    accs.append(acc)
                    t_mrrs.append(t_mrr)
                    t_prec.append(t_prec_set)
            ### iteration turn
            new_train_pairs = []
            lvec, rvec = self.get_pair_embeddings(rest_set_1, rest_set_2)
            A, _, _ = eval_alignment_by_embedding(lvec, rvec, num_threads=self.config.eval_threads)
            B, _, _ = eval_alignment_by_embedding(rvec, lvec, num_threads=self.config.eval_threads)
            A, B = sorted(list(A)), sorted(list(B))
            for a, b in A:
                if B[b][1] == a:
                    new_train_pairs.append([rest_set_1[a], rest_set_2[b]])

            train_pairs = np.concatenate([train_pairs, np.array(new_train_pairs)], axis=0)
            for e1, e2 in new_train_pairs:
                if e1 in rest_set_1:
                    rest_set_1.remove(e1)
                if e2 in rest_set_2:
                    rest_set_2.remove(e2)
        
        return losses, t_prec, accs, t_mrrs, best_acc, best_mrr
                
    
    def evaluate(self, dev_alignments:np.ndarray, hit_k=[1, 5, 10], num_threads:int=16, csls:int=10):
        self.model.eval()
        with torch.no_grad():
            feat_join = self.model()
            feat = feat_join[dev_alignments]
            lvec, rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
            lvec = lvec / np.linalg.norm(lvec, axis=-1, keepdims=True)
            rvec = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
            t_prec_set, acc, t_mrr = eval_alignment_by_embedding(lvec, rvec, hit_k, num_threads, csls, output=True)
        return t_prec_set, acc, t_mrr
    
    def update_entity_embeddings(self) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model()
            self.entity_embeddings = embeddings.detach().cpu().numpy()
        return self.entity_embeddings

    def get_pair_embeddings(self, l_ind, r_ind) -> np.ndarray:
        embeddings = self.update_entity_embeddings()
        lvec, r_vec = embeddings[l_ind], embeddings[r_ind]
        lvec = lvec / np.linalg.norm(lvec, axis=-1, keepdims=True)
        rvec = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
        return lvec, r_vec

    def __get_align_loss(self, pairs:np.ndarray, emb:torch.Tensor, gamma:float=1.0):
        def squared_dist(A:torch.Tensor, B:torch.Tensor):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())
        
        tensor_pairs = torch.from_numpy(pairs).to(self.config.device)
        l, r = tensor_pairs[:, 0].long(), tensor_pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)
        del l_emb, r_emb

        node_size = emb.shape[0]
        mask = 1 - F.one_hot(l, num_classes=node_size) - F.one_hot(r, num_classes=node_size)
        l_loss = (pos_dis - l_neg_dis + gamma) * mask
        r_loss = (pos_dis - r_neg_dis + gamma) * mask
        del l_neg_dis, r_neg_dis

        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()
        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        
        lamb, tau = 20, 8
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        
        return torch.mean(l_loss + r_loss)

    def __get_train_batches(self, train_alignments:np.ndarray, batch_size:int):
        np.random.shuffle(train_alignments)
        batch_num = math.ceil(len(train_alignments)/batch_size)
        train_batches = [train_alignments[i*batch_size : (i+1)*batch_size] for i in range(batch_num)]
        return train_batches
        
    def __get_topk_indices(self, M:torch.Tensor, K:int=1000):
        H, W = M.shape
        M_view = M.view(-1)
        vals, indices = M_view.topk(K)
        print(f"highest sim : {vals[0].item()} , lowest sim : {vals[-1].item()}")
        two_d_indices = torch.cat(((indices//W).unsqueeze(1), (indices%W).unsqueeze(1)), dim=1)
        return two_d_indices, vals

    def __xgea_data_process(self):
        adj_matrix = np.stack(self.loader.adj_matrix.nonzero(), axis=1)
        ent_matrix = np.stack(self.loader.adj_features.nonzero(), axis=1)
        rel_matrix = np.stack(self.loader.rel_features.nonzero(), axis=1)

        if self.config.use_literal:
            features = self.processor.name_embeddings
        else:
            features = self.processor.image_embeddings
        features = F.normalize(torch.Tensor(features).to(self.config.device))
        print(f"XGEA features shape : {features.shape}")
        l_ent, r_ent = self.loader.get_KG_entities(1), self.loader.get_KG_entities(2)
        feat_sim = features[l_ent].mm(features[r_ent].T)
        two_d_indices, _ = self.__get_topk_indices(feat_sim, self.config.unsup_k*100)
        del feat_sim

        count = 0
        visual_links, used_inds = [], []
        for ind in two_d_indices:
            if l_ent[ind[0]] in used_inds or r_ent[ind[1]] in used_inds:
                continue
            used_inds.append(l_ent[ind[0]])
            used_inds.append(r_ent[ind[1]])
            visual_links.append((l_ent[ind[0]], r_ent[ind[1]]))
            count += 1
            if count == self.config.unsup_k:
                break
        for ind in enumerate(self.loader.sup_pairs):
            visual_links.append((ind[0], ind[1]))

        train_pairs = np.array(visual_links, dtype=np.float32)
        return train_pairs, adj_matrix, ent_matrix, rel_matrix
        


class XGEA(nn.Module):
    def __init__(self, 
        node_hidden:int, rel_hidden:int,
        node_size:int, rel_size:int, triple_size:int,
        adj_matrix:torch.Tensor,
        ent_matrix:torch.Tensor,
        rel_matrix:torch.Tensor,
        r_index:torch.Tensor,
        r_val:torch.Tensor,
        features:torch.Tensor,
        dropout_rate:float=0.0,
        depth:int=2,
        device:str="cpu"
    ):
        super(XGEA, self).__init__()

        self.node_hidden = node_hidden
        self.rel_hidden = rel_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device

        self.dropout = nn.Dropout(dropout_rate)

        self.adj_list = adj_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.info_features = features.to(device)

        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        self.info_embedding = nn.Embedding(features.shape[-1], node_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        torch.nn.init.xavier_uniform_(self.info_embedding.weight)

        self.e_encoder = NR_GraphAttention(
            node_size=self.node_size, 
            rel_size=self.rel_size, 
            triple_size=self.triple_size, 
            node_dim=self.node_hidden, 
            depth=self.depth
        )
        self.e_encoder_info = NR_GraphAttention(
            node_size=self.node_size,
            rel_size=self.rel_size,
            triple_size=self.triple_size,
            node_dim=self.node_hidden,
            depth=self.depth
        )
        self.r_encoder = NR_GraphAttention(
            node_size=self.node_size, 
            rel_size=self.rel_size, 
            triple_size=self.triple_size, 
            node_dim=self.node_hidden, 
            depth=self.depth
        )
    
    def avg(self, adj, emb, size:int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float), size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def forward(self):
        # [Ne x Ne] · [Ne x dim] = [Ne x dim]
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        # [Ne x Nr] · [Nr x dim] = [Ne x dim]
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)

        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
        out_rel_feature = self.r_encoder([rel_feature] + opt)
        out_ent_feature = self.e_encoder([ent_feature] + opt)

        out_feature_join = torch.cat([out_ent_feature, out_rel_feature], dim=-1)
        out_feature_join = self.dropout(out_feature_join)

        return out_feature_join


class NR_GraphAttention(nn.Module):
    def __init__(self, 
        node_size, 
        rel_size,
        triple_size,
        node_dim, 
        depth=1
    ):
        super(NR_GraphAttention, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        # attention kernel
        for _ in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        features, rel_emb, adj, r_index, r_val = inputs[:5]
        features = self.activation(features)
        outputs = [features]

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val, size=[self.triple_size, self.rel_size], dtype=torch.float32)   # shape = [N_tri, dim]
            tri_rel = torch.sparse.mm(tri_rel, rel_emb) # shape = [N_tri, dim]
            neighs = features[adj[1, :].long()]
            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att:torch.Tensor = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs*torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0, index=adj[0, :].long())
            features = self.activation(new_features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)
        return outputs
