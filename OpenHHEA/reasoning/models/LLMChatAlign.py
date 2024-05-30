import re
import os
import time
import random
import openai
import numpy as np
from tqdm import tqdm

from OpenHHEA.configs.types import SimDimension
from OpenHHEA.configs.model_configs import LLMChatAlignConfig, DEFAULT_WEIGHTS
from OpenHHEA.data import KGDataLoader
from OpenHHEA.process import KGDataProcessor
from OpenHHEA.evaluate import eval_alignment_by_ranks
from OpenHHEA.reasoning.utils import get_noise_embeddings, generate_candidates_by_sim_mat
from OpenHHEA.reasoning.results import LLMChatAlignResult

### functions of querying LLM
def get_llm_response(llm:openai.OpenAI, model:str, prompt:str, history:list=[], print_log=False):
    messages = history + [{"role": "user", "content": prompt}]
    if print_log:
        print("### PROMPT :")
        p = "\n".join(m["content"] if m["role"] != "system" else "" for m in messages)
        print(p)
    
    response, get_res = None, False
    try:
        response = llm.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens=700,
            temperature=0.3
        )
        get_res = True
    except Exception as e:
        if print_log:
            print(f"### ERROR : {e}")
    if print_log:
        if get_res:
            print("### RESPONSE :")
            print(response.choices[0].message.content)
        else:
            print("### RESPONSE : [ERROR]")
    return response, get_res



class ChatAlignEntity:
    def __init__(self, ent_id:int, name:str, description:str="", neighbors:list=[]):
        self.ent_id = ent_id
        self.name = name
        self.desc = description
        self.neighbors = neighbors
    
    def sample_neighbors(self, neighbor_num:int):
        neighbor_num = len(self.neighbors) if neighbor_num == 0 else neighbor_num
        if len(self.neighbors) > neighbor_num:
            random.shuffle(self.neighbors)
            self.neighbors = self.neighbors[:neighbor_num]


### Main Pipleline of LLMChatAlign

class PipelineLLMChatAlign:
    def __init__(self, config:LLMChatAlignConfig, dataloader:KGDataLoader, dataprocessor:KGDataProcessor):
        self.config = config
        self.loader = dataloader
        self.processor = dataprocessor

        self.llm = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )
        self.model = LLMChatAlign(
            ### LLM configs
            LLM=self.llm,
            api_model=config.api_model,
            ### hypermeters
            history_len=config.history_len,
            candidates_num=config.candidates_num,
            threshold=config.threshold,
            search_range=config.search_range,
            weights=config.weights,
            ### ablation configs
            use_code=config.use_code,
            use_desc=config.use_desc,
            use_name=config.use_name,
            use_struct=config.use_struct,
            use_img=config.use_img,
            use_time=config.use_time,
            ### others
            print_log=config.print_log
        )

        self.entities = self.__get_entity_neighbors()
        self.candidates = self.__get_candidate_entities()

    
    def run(self, save_result_path:str=None):
        ### sample candidates
        sample_candidates, _, _ = self.__sample_candidates(self.candidates, self.config.data_sample_num)
        ### get description
        self.__get_entity_description(sample_candidates)
        ### start rerank
        result = LLMChatAlignResult(
            config=self.config,
            iteration_stat=[0 for _ in range(len(self.config.search_range) + 1)]
        )
        embed_ranks, llm_ranks = [], []
        for eid, cand_data in sample_candidates:
            main_entity = self.entities[eid]
            candidate_entities = []
            for i, cid in enumerate(cand_data["candidates"]):
                cand_ent = self.entities[cid]
                sims = {"embed_sim": cand_data["cand_sims"][i]}
                if self.config.use_img:
                    sims["mmea_sim"] = cand_data["mmea_sims"][i]
                candidate_entities.append([cand_ent, sims])
            rank, iterations, token_count, time_cost = self.model.reasoning_rethinking_loop(
                main_entity=main_entity,
                candidate_entities=candidate_entities,
                ref_ent_id=self.loader.get_pair_ent(eid),
                base_rank=cand_data["ground_rank"]
            )
            embed_ranks.append(cand_data["ground_rank"])
            llm_ranks.append(rank)
            result.update_iteration(iterations)
            result.update_tokens_count(token_count)
            result.update_time_cost(time_cost)
        ### evaluation
        result.embed_hits, result.embed_mrr = eval_alignment_by_ranks(embed_ranks, hit_k=self.config.hit_k)
        result.llm_hits, result.llm_mrr = eval_alignment_by_ranks(llm_ranks, hit_k=self.config.hit_k)

        if save_result_path is not None:
            save_dir = os.path.split(save_result_path)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_result_path, "a+", encoding="utf-8") as fw:
                fw.write("\n" + str(result) + "\n")
        
        return result



    def __get_entity_description(self, candidates:dict):
        ent_ids = set()
        for eid, cand_data in candidates.items():
            ent_ids.add(eid)
            ent_ids.update(cand_data["candidates"])
        ent_ids = list(ent_ids)
        for eid in tqdm(ent_ids, desc="Get Desc"):
            ### construct prompt
            entity = self.entities[eid]
            tuples = [f"[{', '.join(neigh)}]" for neigh in entity.neighbors]
            prompt = f"Given following informations: 1.[Entity] {entity.name}; 2.[Knowledge Tuples] = [{', '.join(tuples)}]. Please answer the question: \n"
            prompt += f"[Question]: What is {entity.name}? Please give a two-sentence brief introduction. The first sentence is to simply describe what is {entity.name}. The second sentence is to give additional description about {entity.name} based on [Knowledge Tuples] and YOUR OWN KNOWLEDGE. Give [answer] strictly in format: [ENT] is ......\n[answer]: "
            ### query LLM
            res, get_res = get_llm_response(self.llm, self.config.api_model,prompt)
            desc = res.choices[0].message.content if get_res else ""
            if "[ENT] is " in desc:
                desc = desc.replace("[ENT]", entity.name)
            self.entities[eid].desc = desc


    def __sample_candidates(self, candidates:dict, sample_num:int=0):
        hit_k = [1, 5, 10]
        selected_datas = [[], [], [], []]
        all_datas = [(eid, data) for eid, data in candidates.items()]
        
        base_ranks = []
        for eid, data in all_datas:
            rank = data["ground_rank"]
            base_ranks.append(rank)
            if rank == 0:
                selected_datas[0].append((eid, data))
            else:
                if rank < 5:
                    selected_datas[1].append((eid, data))
                else:
                    if rank < 10:
                        selected_datas[2].append((eid, data))
                    else:
                        selected_datas[3].append((eid, data))
        hits, mrr = eval_alignment_by_ranks(base_ranks, hit_k)

        num = [int(sample_num * hits[0])]
        for i in range(len(hits) - 1):
            num.append(int(sample_num * hits[i+1]) - sum(num))
        num.append(sample_num - sum(num))
        sample_datas = []
        for i in range(len(num)):
            sample_datas += random.sample(selected_datas[i], num[i])
        
        sample_ranks = [d["ground_rank"] for _, d in sample_datas]
        hits, mrr = eval_alignment_by_ranks(sample_ranks, hit_k=self.config.hit_k)
        return sample_datas, hits, mrr

    
    def __get_candidate_entities(self):
        l_ent, r_ent = self.loader.ref_pairs.T[0], self.loader.ref_pairs.T[1]
        if self.config.use_img:
            img_feats = get_noise_embeddings(self.processor.image_embeddings, self.config.noise_ratio)
            txt_feats = get_noise_embeddings(self.processor.name_embeddings, self.config.noise_ratio)
            ### image-image similarity
            embed1, embed2 = img_feats[l_ent], img_feats[r_ent]
            i2i_sim = np.matmul(embed1, embed2.T)
            ### text-image similarity
            embed1, embed2 = txt_feats[l_ent], img_feats[r_ent]
            t2i_sim = np.matmul(embed1, embed2.T)
            ### image-text similarity
            embed1, embed2 = img_feats[l_ent], txt_feats[r_ent]
            i2t_sim = np.matmul(embed1, embed2.T)
            ## get multi modal similarity = max(i2i, t2i, i2t)
            mmea_sim:np.ndarray = np.max([i2i_sim, t2i_sim, i2t_sim], axis=0)
        
        ent_feats = get_noise_embeddings(self.processor.entity_embeddings, self.config.noise_ratio)
        embed1, embed2 = ent_feats[l_ent], ent_feats[r_ent]
        ent_sim:np.ndarray = np.matmul(embed1, embed2.T)

        if self.config.use_img:
            sim = (1 - self.config.alpha) * ent_sim + self.config.alpha * mmea_sim
        else:
            sim = ent_sim
        candidates = generate_candidates_by_sim_mat(sim, l_ent, r_ent, cand_num=self.config.candidates_num)

        if self.config.use_img:
            l_e2i = {e:i for i, e in enumerate(l_ent)}
            r_e2i = {e:i for i, e in enumerate(r_ent)}
            for e1, candidate_data in candidates.items():
                mmea_sims = []
                for e2 in candidate_data["candidates"]:
                    id1, id2 = l_e2i[e1], r_e2i[e2]
                    mmea_sims.append(float(mmea_sim[id1][id2]))
                candidates[e1]["mmea_sims"] = mmea_sims

        return candidates

    def __get_entity_neighbors(self):
        entities:dict[int, ChatAlignEntity] = {}
        for eid in self.loader.get_all_entities():
            if self.config.use_name:
                name = self.loader.get_name_by_eid(eid)
            else:
                name = str(eid)
            entities[eid] = ChatAlignEntity(
                ent_id=eid,
                name=name
            )

        for triple in self.loader.get_all_triples():
            h, r, t = triple[:3]
            if self.loader.ent_has_name(h) and self.loader.ent_has_name(t) and self.loader.rel_has_name(r):
                if self.config.use_time:
                    neighbor = triple
                else:
                    neighbor = (h, r, t)
                entities[h].neighbors.append(neighbor)
                entities[t].neighbors.append(neighbor)
        
        for eid in entities.keys():
            entities[eid].sample_neighbors(self.config.neighbor_num)
        return entities


### Model of LLMChatAlign
class LLMChatAlign:
    def __init__(self,
        ### LLM
        LLM:openai.OpenAI,
        api_model:str,
        ### hyperparamters
        history_len:int=3,
        candidates_num:int=20,
        threshold:float=0.5,
        search_range:list[int]=[1, 10, 20],
        weights:dict[SimDimension, float]=DEFAULT_WEIGHTS,
        ### ablation study
        use_code:bool=True,
        use_desc:bool=True,
        use_name:bool=True,
        use_struct:bool=True,
        use_img:bool=True,
        use_time:bool=True,
        print_log:bool=True
    ):
        self.print_log=print_log
        self.llm = LLM
        self.api_model = api_model

        self.history_len = history_len
        self.threshold = threshold
        self.search_range = search_range
        self.candidates_num = candidates_num

        self.use_code = use_code
        self.use_desc = use_desc
        self.use_name = use_name
        self.use_struct = use_struct
        self.use_img = use_img
        self.use_time = use_time
        
        self.info, self.fields, self.weights, self.format_text, self.output_format_list = self.init_sim_dimension(weights)
        self.system_prompt = self.init_prompt()


    ### main API
    def reasoning_rethinking_loop(self, main_entity:ChatAlignEntity, candidate_entities:list[tuple[ChatAlignEntity, dict[str, float]]], ref_ent_id:int, base_rank:int):
        '''
        the loop of reasoning and rethinking by LLM
        Paramters
        ------------
        main_entity     : ChatAlignEntity
            the main entity to be realigned
        candidate_entities  : list[tuple[ChatAlignEntity, float]]
            the candidate entities to be aligned with main entity, each element in list is in format (candidate_entity:ChatAlignEntity, {"embed_sim": sim_score, "mmea_sim": sim_score, ...})
        ref_ent_id      : int
            the entity which is truly aligned with main entity in ref_pairs
        base_rank       : int
            rank of ref_ent_id from embedding-based methods

        Return
        ------------
        output  : (int, int, list[int], float)
            rank        : int, the rank of ref_ent
            iterations  : int, the iteration amount used
            tokens_count    : list[int], statistics on the amount of tokens used by LLM
            time_cost   : float, the time cost of reasoning-rethinking loop 
        '''
        st = time.time()
        rank = base_rank
        iterations = 0
        tokens_count = []

        ### accelerate evaluation process by using some rules
        if base_rank >= self.candidates_num:
            return rank, 0, tokens_count, time.time() - st
        base_sims = [sim_score["embed_sim"] for _, sim_score in candidate_entities]
        if base_sims[0] - base_sims[1] > self.threshold:
            return rank, 0, tokens_count, time.time() - st

        ### start loop
        chat_history = []
        candidate_scores = {cand.ent_id:[str(cand.ent_id), cand.name, 0.0] for cand, _ in candidate_entities}
        cand_list = [candidate for candidate, _ in candidate_entities]
        ranked_candidate_list = []

        while iterations < len(self.search_range):
            ### reasoning by LLM
            responses:list[tuple[ChatAlignEntity, float]] = []
            if self.print_log:
                print(f"### ITERATIONS {iterations + 1}")
            for candidate, sim_score in list(reversed(candidate_entities[:self.search_range[iterations]])):
                prompt, simple_prompt = self.generate_prompt(main_entity, candidate, cand_list, ranked_candidate_list)
                res, get_res = get_llm_response(self.llm, self.api_model, prompt, [{"role": "system", "content": self.system_prompt}] + chat_history, print_log=self.print_log)

                if get_res:
                    response_content = res.choices[0].message.content
                    tokens_count.append(res.usage.total_tokens)
                    ###### extract similarity score and calculate the final score
                    sims = [self.__get_score(response_content, f) for f in self.fields]
                    score = 0.0
                    for j in range(len(sims)):
                        score += self.weights[j] * (sims[j] - 1) * 0.25
                    if self.use_img:
                        img_sim = sim_score["mmea_sim"]
                        if img_sim > 0:
                            score = (1 - self.weights[-1]) * score + self.weights[-1] * img_sim
                    score = round(score, 5)

                    # update chat history
                    if len(chat_history) >= self.history_len * 2:
                        chat_history = chat_history[2:]
                    simple_think_step = []
                    for i, s in enumerate(self.format_text):
                        simple_think_step.append(f"{s} = {sims[i]} out of 5")
                    simple_response = f"{', '.join(simple_think_step)}."
                    chat_history = chat_history + [
                        {"role": "user", "content": simple_prompt},
                        {"role": "assistant", "content": simple_response}
                    ]
                    if self.print_log:
                        sim_name = ["[VERY LOW]", "[LOW]", "[MEDIUM]", "[HIGH]", "[VERY HIGH]"]
                        program_output = f"### PROGRAM: {main_entity['name']} && {candidate['name']}, "
                        for i, f in enumerate(self.format_text):
                            program_output += f"{f}: {sims[i]}-{sim_name[sims[i]-1]}, "
                        if self.use_img:
                            program_output += f"[PROBABILITY OF IMAGES POINTING SAME ENTITY]: {img_sim:.4f}, "
                        program_output += f"FINAL SCORE : {score:.3f} , has ranked entity num: {len(ranked_candidate_list)}"
                        print(program_output)
                    
                    # update ranked candidate list
                    candidate_scores[candidate.ent_id][2] = score
                    ranked_candidate_list = []
                    for rank_cand in candidate_scores.values():
                        if rank_cand[2] != 0.0:
                            name = rank_cand[1] if self.use_name else rank_cand[0]
                            ranked_candidate_list.append((name, rank_cand[2]))
                    ranked_candidate_list = sorted(ranked_candidate_list, key=lambda x:x[1], reverse=True)

                    responses.append((candidate, score))

                if self.print_log:
                    print("###############################################")
            
            iterations += 1
            ### rethinking
            aligned_pairs:list[tuple[ChatAlignEntity, float, int]] = []
            if len(responses) > 0:
                sorted_entities = sorted(responses, key=lambda x:x[1], reverse=True)
                ### update rank of ref_ent
                for j, (cand, score) in enumerate(sorted_entities):
                    aligned_pairs.append((cand, score, j))
                    if cand.ent_id == ref_ent_id:
                        rank = j
                if self.print_log:
                    print([(cand.ent_id, cand.name, score) for cand, score, _ in aligned_pairs])
                ### rethinking by LLM; also accelerate this process by some rules
                if aligned_pairs[0][1] < 0.5 or (len(aligned_pairs) == 1 and aligned_pairs[0][1] < 0.7):    ###### score is too low, go to next iteration
                    good_enough = False
                elif aligned_pairs[0][1] > 0.8: ##### score is good enough, stop loop
                    good_enough = True
                else:
                    if len(aligned_pairs) > 1 and aligned_pairs[0][1] - aligned_pairs[1][1] > 0.5:
                        ##### score of top-rank entity is far higher than others, which means good enough, stop loop
                        good_enough = True
                    else:
                        good_enough, use_tokens = self.ask_for_accuracy(main_entity, aligned_pairs, cand_list)
                        tokens_count.append(use_tokens)
                if good_enough:
                    break
        
        return rank, iterations, tokens_count, time.time()-st


    ### functions of initializing
    def init_sim_dimension(self, weights:dict[SimDimension, float]):
        info, fields, new_weights, format_text = [], [], [], []
        if self.use_name:
            info.append("name")
            fields.append("NAME SIMILARITY")
            format_text.append("[NAME SIMILARITY]")
            new_weights.append(weights[SimDimension.NAME] if SimDimension.NAME in weights else DEFAULT_WEIGHTS[SimDimension.NAME])
        if self.use_name and self.use_desc:
            info.append("description")
            fields.append("PROBABILITY OF DESCRIPTION POINTING SAME ENTITY")
            format_text.append("[PROBABILITY OF DESCRIPTION POINTING SAME ENTITY]")
            new_weights.append(weights[SimDimension.DESC] if SimDimension.DESC in weights else DEFAULT_WEIGHTS[SimDimension.DESC])
        if self.use_struct:
            info.append("structure")
            fields.append("STRUCTURE SIMILARITY")
            format_text.append("[STRUCTURE SIMILARITY]")
            new_weights.append(weights[SimDimension.STRUCT] if SimDimension.STRUCT in weights else DEFAULT_WEIGHTS[SimDimension.STRUCT])
        if self.use_time:
            info.append("time")
            fields.append("TIME SIMILARITY")
            format_text.append("[TIME SIMILARITY]")
            new_weights.append(weights[SimDimension.TIME] if SimDimension.TIME in weights else DEFAULT_WEIGHTS[SimDimension.TIME])
        info.append("YOUR OWN KNOWLEDGE")
        new_weights = [w/sum(new_weights) for w in new_weights]
        if self.use_img:
            new_weights.append(weights[SimDimension.IMAGE] if SimDimension.IMAGE in weights else DEFAULT_WEIGHTS[SimDimension.IMAGE])
        output_format_list = []
        for i, f in enumerate(format_text):
            output_format_list.append(f"{f} = {chr(i+ord('A'))} out of 5")
        return info, fields, new_weights, format_text, output_format_list
    

    def init_prompt(self):
        '''
        Initialize System Prompt of LLM, including KG2Code, Reasoning Steps, Reasnoning Examples and Output Format Restrictions
        Return
        ------------
        output  : str
            the system prompt
        '''
        prompt = ""
        if self.use_code:   ### KG2Code
            tuple_format = 'head_entity, relation, tail_entity'
            if self.use_time:
                tuple_format += ', start_time, end_time'
            # entity class definition
            class_init_head = "Class Entity: def __init__(self, ent_id"
            if self.use_name:
                class_init_head += ", name"
            if self.use_name and self.use_desc:
                class_init_head += ", description"
            if self.use_struct:
                class_init_head += ", tuple=[]"
            class_init_head += "): "
            class_init_body = "self.entity_id = ent_id"
            if self.use_name:
                class_init_body += "self.entity_name = name; "
            if self.use_name and self.use_desc:
                class_init_body += "self.entity_description = description; "
            if self.use_struct:
                class_init_body += "self.tuples = tuples; "
            class_init = f"A Knowledge Graph Entity is defined as follows: " + class_init_head + class_init_body
            # function definition
            func_get_neighbors = f" def get_neighbors(self): neighbors = set(); for {tuple_format} in self.tuples: if head_entity == {'self.entity_name' if self.use_name else 'self.entity_id'}: neighbors.add(tail_entity); else: neighbors.add(head_entity); return list(neighbors)" if self.use_struct else ""
            func_get_relation_information = f" def get_relation_information(self): relation_info = []; for {tuple_format} in self.tuples: relation_info.append(relation); return relation_info" if self.use_struct else ""
            func_get_time_information = f" def get_time_information(self): time_info = []; for {tuple_format} in self.tuples: time_info.append((start_time, end_time)); return time_info;" if self.use_time else ""
            prompt = class_init + func_get_neighbors + func_get_relation_information + func_get_time_information + "\n "
        ### basic role and task
        used_information = []
        if self.use_name:
            used_information.append(f"name information{' (self.entity_name)' if self.use_code else ''}")
        if self.use_name and self.use_desc:
            used_information.append(f"description information{' (self.entity_description)' if self.use_code else ''}")
        if self.use_struct:
            used_information.append(f"structure information{' (self.tuples, get_neighbors(), get_relation_information())' if self.use_code else ''}")
        if self.use_time:
            used_information.append(f"time information{' (get_time_information())' if self.use_code else ''}")
        used_information.append(f"YOUR OWN KNOWLEDGE")
        prompt += f"You are a helpful assistant, helping me align or match entities of knowledge graphs according to {', '.join(used_information)}.\n "
        ### reasoning example
        example = "Your reasoning process for entity alignment should strictly follow this case step by step: "
        example_entity_1 = {
            "ent_id": "'8535'",
            "name": "'Fudan University'",
            "desc": "'Fudan University, Located in Shanghai, established in 1905, is a prestigious Chinese university known for its comprehensive academic programs and membership in the elite C9 League.'",
            "tuples": "[(Fudan University, Make Statement, China, 2005-11, 2005-11), (Vincent C. Siew, Express intent to meet or negotiate, Fudan University, 2001-05, 2001-05), (Fudan University, Make an appeal or request, Hong Kong, 2003-09, 2003-09)]"
        }
        example_entity_2 = {
            "ent_id": "'24431'",
            "name": "'Fudan_University'",
            "desc": "'Fudan_University in Shanghai, founded in 1905, is a top-ranked institution in China, renowned for its wide range of disciplines and part of the C9 League.'",
            "tuples": "[(Fudan_University, country, China, ~, ~), (Fudan_University, instance of, University, ~, ~), (Shoucheng_Zhang, educated at, Fudan_University, ~, ~)]"
        }
        if self.use_code:
            example_def_1 = f"Entity({example_entity_1['ent_id']}{', ' + example_entity_1['name'] if self.use_name else ''}{', ' + example_entity_1['desc'] if self.use_name and self.use_desc else ''}{', ' + example_entity_1['tuples'] if self.use_struct else ''})"
            example_def_2 = f"Entity({example_entity_2['ent_id']}{', ' + example_entity_2['name'] if self.use_name else ''}{', ' + example_entity_2['desc'] if self.use_name and self.use_desc else ''}{', ' + example_entity_2['tuples'] if self.use_struct else ''})"
            example += f"Given [Main Entity] l_e = {example_def_1}, and [Candidate Entity] r_e = {example_def_2}. "
        else:
            example_def_1 = f"entity ID is {example_entity_1['ent_id']}{(', name is ' + example_entity_1['name']) if self.use_name else ''}{(', description is ' + example_entity_1['desc']) if self.use_name and self.use_desc else ''}{(', related knowledge tuples are ' + example_entity_1['tuples'] if self.use_struct else '')}"
            example_def_2 = f"entity ID is {example_entity_2['ent_id']}{(', name is ' + example_entity_2['name']) if self.use_name else ''}{(', description is ' + example_entity_2['desc']) if self.use_name and self.use_desc else ''}{(', related knowledge tuples are ' + example_entity_2['tuples'] if self.use_struct else '')}"
            example += f"Given [Main Entity], whose {example_def_1}; and [Candidate Entity], whose {example_def_2}. "
        example += f" - Do [Main Entity] and [Candidate Entity] align or match? You need to think of the answer STEP BY STEP with {', '.join(self.info)}: "
        example_step_name = f"think of [NAME SIMILARITY] using name information{' (self.entity_name)' if self.use_code else ''} : 'Fudan University' and 'Fudan_University' are almost the same from the string itself and its semantic information with only a slight difference, so [NAME SIMILARITY] = 5 out of 5, which means name similarity is [VERY HIGH]"
        example_step_description = f"think of [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] using description information{' (self.entity_description)' if self.use_code else ''}: the two description all point the same entity, Fudan University in Shanghai, a top-ranked university in China, so [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = 5 out of 5, which means the probability is [VERY HIGH]"
        example_step_struct = f"think of [STRUCTURE SIMILARITY] using tuples information{' (self.tuples, get_neighbors() and get_relation_information())' if self.use_code else ''} : you can find that China is the common neighbor of l_e and r_e, so [STRUCTURE SIMILARITY] = 3 out of 5, which means [STRUCTURE SIMILARITY] is [MEDIUM]"
        example_step_time = f"think of [TIME SIMILARITY] using temporal information{' (get_time_information())' if self.use_code else ''} : you can find that r_e does not have specific time information, so just assume [TIME SIMILARITY] = 2 out of 5, which means [TIME SIMILARITY] is [LOW]"
        example_step = []
        if self.use_name:
            example_step.append(example_step_name)
        if self.use_name and self.use_desc:
            example_step.append(example_step_description)
        if self.use_struct:
            example_step.append(example_step_struct)
        if self.use_time:
            example_step.append(example_step_time)
        for i, step in enumerate(example_step):
            example += f"Step {i+1}, {step}. "
        prompt += example
        ### output format
        output_format = f"\n [Output Format]: {', '.join(self.output_format_list)}. "
        output_format += f"NOTICE, {','.join([chr(i+ord('A')) for i in range(len(self.format_text))])} are in range [1, 2, 3, 4, 5], which respectively means [VERY LOW], [LOW], [MEDIUM], [HIGH], [VERY HIGH]. NOTICE, you MUST strictly output like [Output Format]."
        prompt += output_format

        return prompt

    ### functions for algorithm pipeline
    def generate_prompt(self, main_entity:ChatAlignEntity, cand_entity:ChatAlignEntity, cand_list:list[ChatAlignEntity], ranked_candidate_list:list[tuple[str, float]]=[]):
        '''
        Generate reasoning prompts
        Parameter
        ------------
        main_entity/cand_entity : ChatAlignEntity
            entities to be rematch
        cand_list   : list[ChatAlignEntity]
            list of all candidate entities for main_entity, each element is in type ChatAlignEntity
        ranked_cand_list    : list[tuple[str, float]]
            list of all candidate entities ranked by LLM, in format [(cand0_name, score0), ......, (cand19_name, score19)]

        Return
        -----------
        output  : (str, str)
            two prompts, the first one is complete prompt for reasoning, and the second one is simple prompt just for history record
        '''
        if self.print_log:
            print(f"### GENERATED PROMPT: {main_entity.name} && {cand_entity.name}")

        ### basic entity information
        cand_ent_list = ", ".join([cand.name for cand in cand_list]) if self.use_name else ', '.join([f"'{cand.ent_id}'" for cand in cand_list])
        prompt = f"[Candidate Entities List] which may be aligned with [Main Entity] {main_entity.name if self.use_name else main_entity.ent_id} are shown in the following list: [{cand_ent_list}]. "
        ranked_cand = f"Among [Candidate Entities List], ranked entities are shown as follows in format (candidate, align score): [{', '.join([f'({cand}, {score:.3f})' for cand, score in ranked_candidate_list])}]. " if len(ranked_candidate_list) > 0 else ""
        prompt += ranked_cand

        main_ent_neighbors = ["(" + ", ".join(list(neigh)) + ")" for neigh in main_entity.neighbors]
        cand_ent_neighbors = ["(" + ", ".join(list(neigh)) + ")" for neigh in cand_entity.neighbors]
        entity_1 = {"ent_id": f"'{main_entity.ent_id}'", "name": f"'{main_entity.name}'", "desc": f"'{main_entity.desc}'", "tuples": f"[{', '.join(main_ent_neighbors)}]"}
        entity_2 = {"ent_id": f"'{cand_entity.ent_id}'", "name": f"'{cand_entity.name}'", "desc": f"'{cand_entity.desc}'", "tuples": f"[{', '.join(cand_ent_neighbors)}]"}
        if self.use_code:
            main_ent = f"Entity({entity_1.ent_id}{', ' + entity_1['name'] if self.use_name else ''}{', ' + entity_1['desc'] if self.use_name and self.use_desc else ''}{', ' + entity_1['tuples'] if self.use_struct else ''})"
            cand_ent = f"Entity({entity_2.ent_id}{', ' + entity_2['name'] if self.use_name else ''}{', ' + entity_2['desc'] if self.use_name and self.use_desc else ''}{', ' + entity_2['tuples'] if self.use_struct else ''})"
            prompt += f"\nNow given [Main Entity] l_e = {main_ent}, and [Candidate Entity] r_e = {cand_ent}."
        else:
            main_ent = f"entity ID is {entity_1.ent_id}{(', name is ' + entity_1['name']) if self.use_name else ''}{(', related knowledge tuples are ' + entity_1['tuples'] if self.use_struct else '')}"
            cand_ent = f"entity ID is {entity_2.ent_id}{(', name is ' + entity_2['name']) if self.use_name else ''}{(', related knowledge tuples are ' + entity_2['tuples'] if self.use_struct else '')}"
            prompt += f"\nNow given [Main Entity], whose {main_ent}; and [Candidate Entity], whose {cand_ent}. "

        ### reasoning step
        think_step = f"- Compared with other Candidate Entities, do [Main Entity] and [Candidate Entity] align or match? Think of the answer STEP BY STEP with {', '.join(self.info)}: "
        if self.use_code:
            steps = []
            if self.use_name:
                steps.append("using self.entity_name")
            if self.use_name and self.use_desc:
                steps.append("using self.entity_description")
            if self.use_struct:
                steps.append("using self.tuples, get_neighbors() and get_relation_information()")
            if self.use_time:
                steps.append("using get_time_information()")
        for i, output_format in enumerate(self.output_format_list):
            step = f"think of {output_format}"
            if self.use_code:
                step += f", {steps[i]}"
            think_step += f"Step {i+1}, {step}. "
        if self.use_desc:
            think_step += "NOTICE, the information provided above is not sufficient, so use YOUR OWN KNOWLEDGE to complete them.\n"
        prompt += think_step

        ### output format
        prompt += f" Output answer strictly in format: {', '.join(self.output_format_list)}. "

        ### simple prompt content, remove entity information
        simple_prompt = f"Do [Main Entity] {main_entity.name} and [Candidate Entity] {cand_entity.name} align or match?"
        simple_prompt += f"Think of {', '.join(self.format_text)}."

        return prompt, simple_prompt

    def ask_for_accuracy(self, main_entity:ChatAlignEntity, candidate_pairs:list[tuple[ChatAlignEntity, float, int]], cand_list:list[ChatAlignEntity]):
        '''
        Rethinking Process
        Parameters
        ------------
        main_entity     : ChatAlignEntity
            the main entity to be rerank
        candidate_pairs : list[tuple[ChatAlignEntity, float, int]]
            the pairs need to be rethink, each element in list is in format (cand_entity:ChatAlignEntity, alignment_score:float, alignment_rank:int)
        cand_list       : list[ChatAlignEntity]
            list of all candidate entities for main_entity
        
        Return
        ------------
        output  : bool
            whether to stop the reasoning-rethinking loop
        '''
        cand_ent_list = ", ".join([cand.name for cand in cand_list]) if self.use_name else ", ".join([f"'{cand.ent_id}'" for cand in cand_list])
        prompt = f"[Candidate Entities List] which may be aligned with [Main Entity] {main_entity.name if self.use_name else main_entity.ent_id} are shown in the following list: [{cand_ent_list}]. "
        ### alignment results
        prompt += "Now given the following entity alignments: "
        align_cand_list = []
        for i, candidate in enumerate(candidate_pairs):
            cand_pair_text = f"Candidate {i} = ("
            if self.use_name:
                cand_pair_text += f"name = {candidate[0].name}"
            else:
                cand_pair_text += f"ent_id = {candidate[0].ent_id}"
            cand_pair_text += f", align score={candidate[1]}, rank={candidate[2]})"
            align_cand_list.append(cand_pair_text)
        prompt += f"[Main Entity]: {main_entity.name if self.use_name else main_entity.ent_id} -> [{', '.join(align_cand_list)}]. "
        ### rethinking
        prompt += "Compared with candidate entities in [Candidate Entities List], please answer the question: Do these entity alignments are satisfactory enough ([YES] or [NO])?\n"
        prompt += "Answer [YES] if they are relatively satisfactory, which means the alignment score of the top-ranked candidate meet the threshold, and is far higher than others; otherwise, answer [NO] which means we must search other candidate entities to match with [Main Entity]. "
        prompt += "NOTICE, Just answer [YES] or [NO]. Your reasoning process should follow [EXAMPLE]s:\n"
        ### rethinking examples
        example1 = "1.[EXAMPLE]:\n [user]: Give the following entity alignments: pMain Entity]: Eric Cantor -> [Candidate 0 = (name=Eric_Cantor, align score=0.92, rank=0), Candidate 1 = (name=George_J._Mitchell, align score=0.4, rank=1), Candidate 2 = (name=John_Turner, align score=0.2, rank=2)].\n [reasoning process]: Given this result, you can find that the alignment score of the first candidate in list 'Eric_Cantor', 0.92, is high enough and is far higher than others, 0.4 and 0.2. So the alignment is relatively satisfactory.\n [assistant]: [YES].\n"
        example2 = "2.[EXAMPLE]:\n [user]: Give the following entity alignments: [Main Entity]: Fudan University -> [Candidate 0 = (name=Peking University, align score=0.6, rank=0), Candidate 1 = (name=Tsinghua University, align score=0.5, rank=1), Candidate 2 = (name=Renming University, align score=0.45, rank=2)].\n [reasoning process]: Given this result, you can find that there is another candidate with a score, 0.5 and 0.45, close to the top-ranked candidate's score 0.6, so the search must continue to ensure a more accurate alignment. \n [assistant]: [NO].\n"
        prompt += example1
        prompt += example2
        prompt += "Just directly answer [YES] or [NO], don't give other text. [assistant]:"
        
        ### request LLM
        res, get_res = get_llm_response(self.llm, self.api_model, prompt, print_log=self.print_log)
        ans = False
        if get_res and res is not None:
            res_content = res.choices[0].message.content
            ans = "yes" in res_content.lower()
        return ans, res.usage.total_tokens

    def __get_score(self, res:str, dimension:str):
        '''
        get score from llm's response
        Parameters
        ------------
        res     : str
            the response of LLM
        dimension   : str
            the score dimension, selected from NAME SIMILARIY, PROBABILITY OF DESCRIPTION POINTING SAME ENTITY, STRUCTURE SIMILARITY, TIME SIMILARITY and IMAGE SIMILARITY
        
        Return
        ------------
        output  : int
            the score, ranged in [1, 2, 3, 4, 5]
        '''
        content = res.replace("\n", " ").replace("\t", " ")
        content = re.sub(r"\d{2}\d*", "", content)
        score = 1
        if dimension in content:
            score_find = re.findall(f"{dimension}\D*[=|be|:] \d+", content)
            if len(score_find) > 0:
                score_find = re.findall(f"\d+", score_find[-1])
                score = int(score_find[-1])
        if score < 1:
            if self.print_log:
                print(f"### SCORE ERROR : {score}")
            score = 1
        if score > 5:
            if self.print_log:
                print(f"### SCORE ERROR : {score}")
            score = 5
        return score

