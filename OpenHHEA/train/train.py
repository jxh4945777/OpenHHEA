import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from OpenHHEA.configs.model_configs import EmbeddingBasedConfig
from OpenHHEA.evaluate import eval_alignment_by_embedding
from .loss import HHEALoss



class HHEATrainer:
    def __init__(self, 
        config:EmbeddingBasedConfig,
        loss_fn:HHEALoss=None,
        optimizer:Optimizer=torch.optim.Adam,
        model_save_path:str=None
    ) -> None:
        '''
        Paramters
        ---------
        config  : EmbeddingBasedConfig
            configs for training
        loss_fn  : HHEALoss | customized functions
            the loss function
            should be inherited from HHEALoss and rewrite the forward(pairs, features) function
            or be a function whose input params are pairs(np.ndarray, the test alignments) and features(torch.Tensor, features from model)
            return torch.Tensor
        optimizer  : torch.optim.Optimizer
            training optimizer
        
        Example
        ---------
        >>> trainer = HHEATrainer(
        >>>    config = EmbeddingBasedConfig(),
        >>>    loss_fn = HHEALoss,
        >>>    optimizer = torch.optim.RMSprop
        >>> )
        '''
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        model_dir = os.path.split(self.model_save_path)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.use_iter_train = self.config.iter_turn > 1

    def set_loss_fn(self, loss_fn:HHEALoss):
        if loss_fn is None:
            raise Exception("Can't set trainer's loss_fn to be None")
        self.loss_fn = loss_fn
    def set_optimizer(self, optimizer:Optimizer):
        if optimizer is None:
            raise Exception("Can't set trainer's optimizer to be None")
        self.optimizer = optimizer


    def train(self, model:nn.Module, train_alignments:np.ndarray, dev_alignments:np.ndarray, ent_num:int=0, **kwargs):
        if not self.config.has_set_config:
            raise Exception(f"Model config hasn't been set !")
        if self.loss_fn is None:
            raise Exception("Loss Function hasn't been set!")
        if self.optimizer is None:
            raise Exception("Optimizer hasn't been set!")

        model.to(self.config.device)
        optimizer:Optimizer = self.optimizer(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        print(f"model paramters : {self.__get_model_parameters_num(model)}")
        print(f"Train/Test : {len(train_alignments)} / {len(dev_alignments)}")

        if self.use_iter_train:
            rest_set_1 = [e1 for e1, _ in dev_alignments]
            rest_set_2 = [e2 for _, e2 in dev_alignments]
            np.random.shuffle(rest_set_1)
            np.random.shuffle(rest_set_2)

        losses, t_prec, accs, t_mrrs = [], [], [], []
        best_acc = [0] * len(self.config.hit_k)
        best_mrr = 0
        train_pairs = train_alignments
        for turn in range(self.config.iter_turn):
            print(f"\n### Iteration {turn+1}")
            for i in tqdm(range(self.config.epochs), desc=f"Iteration {turn + 1}"):
                model.train()
                optimizer.zero_grad()
                for pairs in self.__get_train_batches(train_pairs, self.config.batch_size, ent_num, self.config.neg_sample):
                    feat = model(**kwargs)
                    loss:torch.Tensor = self.loss_fn(pairs, feat)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
                ### evaluate
                if (i + 1) % self.config.eval_step == 0:
                    t_prec_set, acc, t_mrr = self.evaluate(model, dev_alignments, **kwargs)
                    if best_acc[0] < acc[0]:
                        best_acc = acc
                        best_mrr = t_mrr
                        if self.model_save_path is not None:
                            torch.save(model.state_dict(), self.model_save_path)
                    print(f"// best results : hits@{self.config.hit_k} = {[round(n, 3) for n in best_acc]} , mrr = {best_mrr:.4f} //")
                    accs.append(acc)
                    t_mrrs.append(t_mrr)
                    t_prec.append(t_prec_set)
            ### iteration train
            if self.use_iter_train:
                new_train_pairs = []
                lvec, rvec = self.__get_embeddings_by_ents(model, [rest_set_1, rest_set_2], **kwargs)
                A, _, _ = eval_alignment_by_embedding(lvec, rvec, num_threads=self.config.eval_threads)
                B, _, _ = eval_alignment_by_embedding(rvec, lvec, num_threads=self.config.eval_threads)
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


    def evaluate(self, model:nn.Module, dev_alignments:np.ndarray, **kwargs):
        model.to(self.config.device)
        model.eval()
        with torch.no_grad():
            feat:torch.Tensor = model(**kwargs)[dev_alignments]
            lvec, rvec = feat[:, 0, :].detach().cpu().numpy(), feat[:, 1, :].detach().cpu().numpy()
            lvec = lvec / np.linalg.norm(lvec, axis=-1, keepdims=True)
            rvec = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
            t_prec_set, acc, t_mrr = eval_alignment_by_embedding(lvec, rvec, self.config.hit_k, self.config.eval_threads, self.config.csls, output=True)
        return t_prec_set, acc, t_mrr
    

    def __get_embeddings_by_ents(self, model:nn.Module, ents:list, **kwargs):
        '''
        Parameters
        ----------
        model  : torch.nn.Module
            the model
        ents  : list[list[int]]
            the list of entities, in format [[eid00, eid01, eid02, ...], [eid10, eid11, eid12, ...], ...]
        **kwargs  :
            input params of model forward
        Return
        ----------
        list_of_embeddings  : list[np.ndarray]
            return the embeddings list, in format [embeddings0, embeddings1, ...]
            embeddingsX = NDArray([embed_eidX0, embed_eidX1, ...])
        '''
        model.eval()
        with torch.no_grad():
            embeddings:torch.Tensor = model(**kwargs)
            embeddings = embeddings.detach().cpu().numpy()
        embed_list = []
        for eids in ents:
            embed = embeddings[eids]
            embed = embed / np.linalg.norm(embed, axis=-1, keepdims=True)
            embed_list.append(embed)
        return embed_list

    def __get_train_batches(self, train_alignments:np.ndarray, batch_size:int, ent_num:int=0, neg_sample:bool=False):
        if neg_sample and ent_num <= 0:
            raise Exception(f"Error occured when get train batches : input ent_num = {ent_num} <= 0 when setting neg_sample = True")
        ### supplementing train datas to a batch
        if batch_size > len(train_alignments):
            repeat_num = int(batch_size / len(train_alignments))
            repeat_train_alignments = np.reshape(np.repeat(np.expand_dims(train_alignments, axis=0), axis=0, repeats=repeat_num), newshape=(-1, 2))
            np.random.shuffle(train_alignments)
            train_alignments = np.concatenate([repeat_train_alignments, train_alignments[:batch_size-repeat_num*len(train_alignments)]], axis=0)
        np.random.shuffle(train_alignments)
        ### negative sampling
        if neg_sample:
            train_alignments = np.concatenate([train_alignments, np.random.randint(0, ent_num, train_alignments.shape)], axis=-1)
        batch_num = math.ceil(len(train_alignments) / batch_size)
        train_batches = [train_alignments[i*batch_size : (i+1)*batch_size] for i in range(batch_num)]
        return train_batches

    def __get_model_parameters_num(self, model:nn.Module):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
