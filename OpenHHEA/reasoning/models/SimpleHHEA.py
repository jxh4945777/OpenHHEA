import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from OpenHHEA.configs.model_configs import SimpleHHEAConfig
from OpenHHEA.data import KGDataLoader
from OpenHHEA.process import KGDataProcessor
from OpenHHEA.evaluate import eval_alignment_by_embedding
from OpenHHEA.reasoning.utils import get_noise_embeddings
from OpenHHEA.reasoning.results import SimpleHHEAResult


### Main Pipeline of SimpleHHEA
class PipelineSimpleHHEA:
    def __init__(self, 
        config:SimpleHHEAConfig, 
        dataloader:KGDataLoader, 
        dataprocessor:KGDataProcessor,
        model_path:str=None
    ):
        self.config = config
        self.loader = dataloader
        self.processor = dataprocessor
        self.entity_embeddings = None

        self.model = Simple_HHEA(
            time_span=1 + 27*13,
            ent_name_emb=get_noise_embeddings(self.processor.name_embeddings, noise_ratio=self.config.name_noise_ratio),
            ent_time_emb=self.processor.time_embeddings,
            ent_struct_emb=self.processor.struct_embeddings,
            use_structure=self.config.use_structure,
            use_time=self.config.use_time,
            emb_size=self.config.emb_size,
            structure_size=self.config.structure_size,
            time_size=self.config.time_size,
            device=self.config.device
        ).to(self.config.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))
            self.update_entity_embeddings()

        self.best_model_path = os.path.join(self.config.model_save_dir, f"best_simplehhea_model_noise_{self.config.name_noise_ratio}.pth")

    def run(self, save_result_path:str=None):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")

        st = time.time()
        print(f"Train/Val : {len(self.loader.sup_pairs)} / {len(self.loader.ref_pairs)}")
        node_size = len(self.loader.get_all_entities())
        train_alignment = self.__get_train_set(self.loader.sup_pairs, node_size, node_size)

        losses, t_prec, accs, t_mrrs, best_acc, best_mrr = self.train(
            train_alignment=train_alignment,
            dev_alignments=self.loader.ref_pairs,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gamma=self.config.gamma,
            hit_k=self.config.hit_k
        )

        self.update_entity_embeddings()

        result = SimpleHHEAResult(
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

    def update_entity_embeddings(self) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            embeddings:torch.Tensor = self.model()
            self.entity_embeddings = embeddings.detach().cpu().numpy()
        return self.entity_embeddings
    
    def __get_train_set(self, train_alignments:np.ndarray, batch_size:int, ent_num:int):
        negative_ratio = math.ceil(batch_size / len(train_alignments))
        train_set = np.reshape(np.repeat(np.expand_dims(train_alignments, axis=0), axis=0, repeats=negative_ratio), newshape=(-1, 2))
        np.random.shuffle(train_set)
        train_set = train_set[:batch_size]
        train_set = np.concatenate([train_set, np.random.randint(0, ent_num, train_set.shape)], axis=-1)
        return train_set

    def __get_model_parameters_num(self):
        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def train(self, train_alignment:np.ndarray, dev_alignments:np.ndarray, epochs=1500, learning_rate=0.01, weight_decay=0.001, gamma=1.0, hit_k=[1, 5, 10]):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print(f"paramters : {self.__get_model_parameters_num()}")

        def l1(ll, rr):
            return torch.sum(torch.abs(ll - rr), axis=-1)
        
        losses, t_prec, accs, t_mrrs = [], [], [], []
        best_acc = [0] * len(hit_k)
        best_mrr = 0
        batch_size = len(train_alignment)
        for i in tqdm(range(epochs), desc="SimpleHHEA Training"):
            ### forward
            self.model.train()
            optimizer.zero_grad()
            feat = self.model()[train_alignment]
            ### loss
            l, r, fl, fr = feat[:, 0, :], feat[:, 1, :], feat[:, 2, :], feat[:, 3, :]
            loss = torch.sum(nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + nn.ReLU()(gamma + l1(l, r) - l1(fl, r))) / batch_size
            ### backward
            losses.append(loss.item())
            loss.backward(retain_graph=True)
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

        return losses, t_prec, accs, t_mrrs, best_acc, best_mrr

    def evaluate(self, dev_alignments:np.ndarray, hit_k=[1, 5, 10], num_threads:int=16, csls:int=10):
        self.model.eval()
        with torch.no_grad():
            feat = self.model()[dev_alignments]
            lvec, rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
            lvec = lvec / np.linalg.norm(lvec, axis=-1, keepdims=True)
            rvec = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
            t_prec_set, acc, t_mrr = eval_alignment_by_embedding(lvec, rvec, hit_k, num_threads, csls, youtput=True)
        return t_prec_set, acc, t_mrr


### Time2Vec Model
class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features, activation="cos"):
        super(CosineActivation, self).__init__()
        self.w0 = Parameter(torch.randn(in_features, 1))
        self.b0 = Parameter(torch.randn(1))
        self.w  = Parameter(torch.randn(in_features, out_features-1))
        self.b  = Parameter(torch.randn(out_features-1))
        self.f  = torch.cos if activation == "cos" else torch.sin

    def forward(self, tau):
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 1)

class Time2Vec(nn.Module):
    def __init__(self, hidden_dim):
        super(Time2Vec, self).__init__()
        self.l = CosineActivation(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = self.l(x)
        x = self.fc(x)
        return x

### Main Model, Simple_HHEA
class Simple_HHEA(nn.Module):
    def __init__(self,
        time_span:int=1+27*13,
        ent_name_emb:np.ndarray=None, 
        ent_time_emb:np.ndarray=None, 
        ent_dw_emb:np.ndarray=None,
        use_structure:bool=True,
        use_time:bool=True,
        emb_size:int=64,
        structure_size:int=8,
        time_size:int=8,
        device:str="cuda:0"
    ):
        super(Simple_HHEA, self).__init__()

        self.device = device
        self.use_structure = use_structure
        self.use_time = use_time

        self.emb_size = emb_size
        self.struct_size = structure_size
        self.time_size = time_size

        linear_size_1 = self.emb_size

        if self.use_time:
            linear_size_1 += self.time_size
            self.ent_time_emb = torch.tensor(ent_time_emb).to(self.device).float()
            self.fc_time_0 = nn.Linear(32, 32)
            self.fc_time = nn.Linear(32, self.time_size)
            self.time2vec = Time2Vec(hidden_dim=32)
            self.time_span = time_span 
            self.time_span_index = torch.tensor(np.array([i for i in range(self.time_span)])).to(self.device).unsqueeze(1).float()

        if self.use_structure:
            linear_size_1 += self.struct_size
            self.ent_dw_emb = torch.tensor(ent_dw_emb).to(self.device).float()
            self.fc_dw_0 = nn.Linear(self.ent_dw_emb.shape[-1], emb_size)
            self.fc_dw = nn.Linear(emb_size, self.struct_size)
        
        self.fc_final = nn.Linear(linear_size_1, emb_size)

        self.ent_name_emb = torch.tensor(ent_name_emb).to(self.device).float()
        self.fc_name_0 = nn.Linear(self.ent_name_emb.shape[-1], emb_size)
        self.fc_name = nn.Linear(emb_size, emb_size)

        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

    def forward(self):
        ent_name_feature = self.fc_name(self.fc_name_0(self.dropout(self.ent_name_emb)))
        features = [ent_name_feature]

        if self.use_time:
            time_span_feature = self.time2vec(self.time_span_index)
            ent_time_feature = torch.mm(self.ent_time_emb, time_span_feature) / self.time_span
            ent_time_feature = self.fc_time(self.fc_time_0(self.dropout(ent_time_feature)))
            features.append(ent_time_feature)
        
        if self.use_structure:
            ent_dw_feature = self.fc_dw(self.fc_dw_0(self.dropout(self.ent_dw_emb)))
            features.append(ent_dw_feature)

        output_feature = torch.cat(features, dim=1)
        output_feature = self.fc_final(output_feature)
        
        return output_feature

