import json
import math
from OpenHHEA.configs.model_configs import ModelConfig



class ReasoningResult:
    def __init__(self, config:ModelConfig, hits:list, mrr:float, time_cost:float=0.0):
        self.config = config
        self.hits = hits
        self.mrr = mrr
        self.time_cost = time_cost

    def to_dict(self):
        result_dict = {
            "configs": self.config.to_dict(), 
            "results": {
                f"hits@{self.config.hit_k}": [round(h, 3) for h in self.hits],
                "MRR": f"{self.mrr:.3f}",
                "time_cost": round(self.time_cost, 3)
            }
        }
        return result_dict

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


class SimpleHHEAResult(ReasoningResult):
    def __init__(self, 
        config, 
        hits:list, 
        mrr:float,
        time_cost:float=0.0
    ):
        '''
        SimpleHHEA Result Parameters
        ------------------------------
        config  : SimpleHHEAConfig
            model condfigs
        time_cost   : float
            the time cost by whole pipeline
        hits    : list
            the hits@K result, length is equal to config.hit_k
        mrr     : float
            the MRR result
        '''
        super().__init__(config, hits, mrr, time_cost)


class XGEAResult(ReasoningResult):
    def __init__(self, 
        config, 
        hits:list, 
        mrr:float,
        time_cost:float=0.0
    ):
        '''
        XGEA Result Parameters
        ------------------------------
        config  : XGEAConfig
            model condfigs
        time_cost   : float
            the time cost by whole pipeline
        hits    : list
            the hits@K result, length is equal to config.hit_k
        mrr     : float
            the MRR result
        '''
        super().__init__(config, hits, mrr, time_cost)


class LLMChatAlignResult(ReasoningResult):
    def __init__(self, 
        config, 
        embed_hits:list=[],
        embed_mrr:float=0.0,
        llm_hits:list=[],
        llm_mrr:float=0.0,
        iteration_stat:list=[],
        tokens_count:list=[],
        time_cost:float=0.0
    ):
        '''
        LLM-ChatAlign Result Paramters
        ------------------------------
        config  : LLMChatAlignConfig
            the model config
        embed_hits  : list
            the hits@K results of embedding-based methods
        embed_mrr   : float
            the MRR results of embedding-based methods
        llm_hits    : list
            the hits@K results of llm-enhanced methods which based on embedding-based results
        llm_mrr     : float
            the MRR results of llm-enhanced methods
        iteration_stat  : list
            the statistic of iterations used by llm-enhanced methods for each entity
        tokens_count    : list
            the tokens used by LLM for each entity
        time_cost       : float
            the time cost by whole pipeline
        '''
        super().__init__(config, llm_hits, llm_mrr, time_cost)
        self.embed_hits = embed_hits
        self.embed_mrr = embed_mrr
        
        # self.llm_hits = llm_hits
        # self.llm_mrr = llm_mrr

        self.iteration_stat = iteration_stat
        self.tokens_count = tokens_count
        
        self.total_entity_num = 0
        self.total_request_num = 0

    def update_iteration(self, iteration:int):
        self.iteration_stat[iteration] += 1
        self.total_entity_num += 1
    def update_tokens_count(self, tokens_count:list):
        self.tokens_count = self.tokens_count + tokens_count
        self.total_request_num += len(tokens_count)
    def update_time_cost(self, time_cost:float):
        self.time_cost += time_cost

    def stat_tokens_count(self, interval:int=200):
        total_tokens_count = sum(self.tokens_count)
        average_per_request = total_tokens_count / self.total_request_num
        average_per_entity = total_tokens_count / self.total_entity_num
        
        min_token_count, max_token_count = min(self.tokens_count), max(self.tokens_count)
        stat_s, stat_e = math.floor(min_token_count / interval), math.ceil(max_token_count / interval)
        token_stat_list = [0 for _ in range(stat_e - stat_s)]
        for t in self.tokens_count:
            token_stat_list[int(t/interval) - stat_s] += 1
        token_stat = {}
        for i, cnt in enumerate(token_stat_list):
            if cnt > 0:
                s = interval*(stat_s + i)
                e = s + interval
                token_stat[f"[ {s:4d} , {e:4d} )"] = f"{cnt} / {self.total_request_num} = {cnt/self.total_request_num:.2%}"
        return total_tokens_count, average_per_entity, average_per_request, token_stat
    def stat_time_cost(self):
        total_time_cost = self.time_cost
        average_per_request = total_time_cost / self.total_request_num
        average_per_entity = total_time_cost / self.total_entity_num
        return total_time_cost, average_per_entity, average_per_request

    def to_dict(self):
        _, _, _, token_stat = self.stat_tokens_count(interval=200)

        return {
            "configs": self.config.to_dict(), 
            "results": {
                "embed_based": {
                    f"hits@{self.config.hit_k}": [round(h, 3) for h in self.embed_hits],
                    "MRR": f"{self.embed_mrr:.3f}"
                },
                "llm_enhance": {
                    f"hits@{self.config.hit_k}": [round(h, 3) for h in self.hits],
                    "MRR": f"{self.mrr:.3f}"
                },
                "iterations": self.iteration_stat,
                "tokens_count": token_stat,
                "time_cost": round(self.time_cost, 3)
            }
        }