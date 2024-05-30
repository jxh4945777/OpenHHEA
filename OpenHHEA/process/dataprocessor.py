import os
import numpy as np

from OpenHHEA.data import KGDataLoader
from .name_process import get_name_processor
from .image_process import get_image_processor
from .structure_process import get_struct_processor
from .utils_dataprocess import load_embeddings_from_paths

from OpenHHEA.configs.types import ProcessName, ProcessImage, ProcessStruct, ProcessEntity
from OpenHHEA.configs.process_configs import *



class KGDataProcessor:
    def __init__(self, 
        dataloader:KGDataLoader,
        name_process_configs=None,
        struct_process_configs=None,
        image_process_configs=None,
        entity_process_configs=None
    ):
        '''
        Processor Parameters
        ---------------------
        dataloader  : KGDataLoader
            the datas of KG
        name_configs    : None | tuple[ProcessName(Enum) , dict|ProcessConfigs]
            configs for name processing. The first element is the type for processing method chosen from Enum class ProcessName; the second element if a dict or ProcessConfigs containing detailed configs
        struct_configs  : None | tuple[ProcessStruct(Enum) , dict|ProcessConfigs]
            similar as name_configs
        image_configs   : None | tuple[ProcessImage(Enum) , dict|ProcessConfigs]
            similar as image_configs
        '''
        self.ent_num = len(dataloader.get_all_entities())
        self.name_embeddings = None
        self.struct_embeddings = None
        self.image_embeddings = None
        self.entity_embeddings = None

        self.process_name_embeddings(name_process_configs, dataloader)
        self.process_struct_embeddings(struct_process_configs, dataloader)
        self.process_image_embeddings(image_process_configs, dataloader)
        self.process_entity_embeddings(entity_process_configs, dataloader)

        if dataloader.has_time:
            self.time_embeddings = self.load_ent_time_matrix([len(dataloader.get_KG_entities(0), dataloader.get_KG_entities(1))], [dataloader.get_KG_triples(0), dataloader.get_KG_triples(1)], dataloader.time_id2name)

    ### API for processing embeddings
    def process_embeddings(self, embed_type:str, process_configs:ProcessConfigs, dataloader:KGDataLoader):
        embed_type = embed_type.lower()
        if embed_type in ["name", "txt", "text", "literal"]:
            self.process_name_embeddings(process_configs, dataloader)
        elif embed_type in ["struct", "structure"]:
            self.process_struct_embeddings(process_configs, dataloader)
        elif embed_type in ["img", "image", "pic", "picture"]:
            self.process_image_embeddings(process_configs, dataloader)
        elif embed_type in ["ent", "entity"]:
            self.process_entity_embeddings(process_configs, dataloader)
        else:
            raise Exception(f"Error occured when process embeddings : there is not embedding type {embed_type} , please selected from [\"name\" , \"structure\" , \"image\" , \"entity\"]")

    def process_name_embeddings(self, process_configs:ProcessConfigs, dataloader:KGDataLoader):
        ### check process configs
        self.name_process_type, self.name_process_configs = self.__process_configs_check(process_configs)
        if self.name_process_type is not None:
            if isinstance(self.name_process_type, str):
                if self.name_process_type not in ProcessName._value2member_map_:
                    raise Exception(f"Error occured when process name embeddings : there is no method \"{self.name_process_type}\" , please selected from {list(ProcessName._value2member_map_.keys())}")
                self.name_process_type = ProcessName._value2member_map_[self.name_process_type]
            elif isinstance(self.name_process_type, ProcessName):
                if self.name_process_type not in ProcessName:
                    raise Exception(f"Error occured when process name embeddings : there is no method {self.name_process_type} , please selected from {list(ProcessName.__members__.values())}")
            else:
                raise Exception(f"Error occured when process name embeddings : please use Str or ProcessName to set method_type , not {type(self.name_process_type)}")
        ### process name embeddings
        self.name_processor = None
        if self.name_process_configs is not None:
            if self.name_process_type == ProcessName.LOAD:
                if self.name_process_configs["embeddings"] is not None:
                    self.name_embeddings = self.name_process_configs["embeddings"]
                else:
                    self.name_embeddings = load_embeddings_from_paths(self.ent_num, self.name_process_configs["embedding_paths"])
            else:
                self.name_processor = get_name_processor(self.name_process_type, **self.name_process_configs)
                self.name_embeddings = self.name_processor.name_process(dataloader)

    def process_struct_embeddings(self, process_configs:ProcessConfigs, dataloader:KGDataLoader):
        ### check process configs
        self.struct_process_type, self.struct_process_configs = self.__process_configs_check(process_configs)
        if self.struct_process_type is not None:
            if isinstance(self.struct_process_type, str):
                if self.struct_process_type not in ProcessStruct._value2member_map_:
                    raise Exception(f"Error occured when process structure embeddings : there is no method \"{self.struct_process_type}\" , please selected from {list(ProcessStruct._value2member_map_.keys())}")
                self.struct_process_type = ProcessStruct._value2member_map_[self.struct_process_type]
            elif isinstance(self.struct_process_type, ProcessStruct):
                if self.struct_process_type not in ProcessStruct:
                    raise Exception(f"Error occured when process structure embeddings : there is no method {self.struct_process_type} , please selected from {list(ProcessStruct.__members__.values())}")
            else:
                raise Exception(f"Error occured when process structure embeddings : please use Str or ProcessStruct to set method_type , not {type(self.struct_process_type)}")
        ### process structure embeddings
        self.struct_processor = None
        if self.struct_process_configs is not None:
            if self.struct_process_type == ProcessStruct.LOAD:
                if self.struct_process_configs["embeddings"] is not None:
                    self.struct_embeddings = self.struct_process_configs["embeddings"]
                else:
                    self.struct_embeddings = load_embeddings_from_paths(self.ent_num, self.struct_process_configs["embedding_paths"])
            else:
                self.struct_processor = get_struct_processor(self.struct_process_type, **self.struct_process_configs)
                self.struct_embeddings = self.struct_processor.struct_process(dataloader)

    def process_image_embeddings(self, process_configs:ProcessConfigs, dataloader:KGDataLoader):
        ### check process configs
        self.image_process_type, self.image_process_configs = self.__process_configs_check(process_configs)
        if self.image_process_type is not None:
            if isinstance(self.image_process_type, str):
                if self.image_process_type not in ProcessImage._value2member_map_:
                    raise Exception(f"Error occured when process image embeddings : there is no method \"{self.image_process_type}\" , please selected from {list(ProcessImage._value2member_map_.keys())}")
                self.image_process_type = ProcessImage._value2member_map_[self.image_process_type]
            elif isinstance(self.image_process_type, ProcessImage):
                if self.image_process_type not in ProcessImage:
                    raise Exception(f"Error occured when process image embeddings : there is no method {self.image_process_type} , please selected from {list(ProcessImage.__members__.values())}")
            else:
                raise Exception(f"Error occured when process image embeddings : please use Str or ProcessImage to set method_type , not {type(self.image_process_type)}")
        ### process image embeddings
        self.image_processor = None
        if self.image_process_configs is not None:
            if self.image_process_type == ProcessImage.LOAD:
                if self.image_process_configs["embeddings"] is not None:
                    self.image_embeddings = self.image_process_configs["embeddings"]
                else:
                    self.image_embeddings = load_embeddings_from_paths(self.ent_num, self.image_process_configs["embedding_paths"])
            else:
                self.image_processor = get_image_processor(self.image_process_type, **self.image_process_configs)
                self.image_embeddings = self.image_processor.image_process(dataloader)

    def process_entity_embeddings(self, process_configs:ProcessConfigs, dataloader:KGDataLoader):
        ### check process configs
        self.entity_process_type, self.entity_process_configs = self.__process_configs_check(process_configs)
        if self.entity_process_type is not None:
            if isinstance(self.entity_process_type, str):
                if self.entity_process_type not in ProcessEntity._value2member_map_:
                    raise Exception(f"Error occured when process entity embeddings : there is no method \"{self.entity_process_type}\" , please selected from {list(ProcessEntity._value2member_map_.keys())}")
                self.entity_process_type = ProcessEntity._value2member_map_[self.entity_process_type]
            elif isinstance(self.entity_process_type, ProcessEntity):
                if self.entity_process_type not in ProcessEntity:
                    raise Exception(f"Error occured when process entity embeddings : there is no method {self.entity_process_type} , please selected from {list(ProcessEntity.__members__.values())}")
            else:
                raise Exception(f"Error occured when process entity embeddings : please use Str or ProcessEntity to set method_type , not {type(self.entity_process_type)}")
        ### process entity embeddings
        self.entity_processor = None
        if self.entity_process_configs is not None:
            if self.entity_process_type == ProcessEntity.LOAD:
                if self.entity_process_configs["embeddings"] is not None:
                    self.entity_embeddings = self.entity_process_configs["embeddings"]
                else:
                    self.entity_embeddings = load_embeddings_from_paths(self.ent_num, self.entity_process_configs["embedding_paths"])
            # else:
            #     self.entity_processor = get_entity_processor(self.entity_process_type, **self.entity_process_configs)
            #     self.entity_embeddings = self.entity_processor.entity_process(dataloader)

    def __process_configs_check(self, process_configs):
        if process_configs is None:
            return None, None
        if isinstance(process_configs, ProcessConfigs):
            return process_configs.method_type, process_configs.to_dict()
        elif isinstance(process_configs, dict):
            return process_configs["method_type"], {k:v for k, v in process_configs.items() if k != "method_type"}
        else:
            raise Exception(f"Error occured when process datas : please input dict or ProcessConfigs to set configs.")

    def load_ent_time_matrix(self, ent_nums, KG_triples, time_dict):
        ent_1_num, ent_2_num = ent_nums
        
        def rel_time_cal(time_year, time_month):
            return (time_year - 1995) * 13 + time_month + 1
        time_emb_size = 1 + 27 * 13
        ent_1_emb = np.zeros([ent_1_num, time_emb_size])
        ent_2_emb = np.zeros([ent_2_num, time_emb_size])
        for triple in KG_triples[0]:
            h, _, _, ts, te = triple
            for tau in [ts, te]:
                time_name = time_dict[tau]
                if time_name != "~":
                    time_y, time_m = [int(t) for t in time_name.split("-")]
                    if time_y < 1995:
                        ent_1_emb[h, 0] += 1
                    else:
                        ent_1_emb[h, rel_time_cal(time_y, time_m)] += 1
        for triple in KG_triples[1]:
            time_y_s, time_m_s = 0, 0
            time_y_e, time_m_e = 0, 0
            h, _, _, ts, te = triple
            time_name_ts, time_name_te = time_dict[ts], time_dict[te]
            if time_name_ts != "~":
                time_y_s, time_m_s = [int(t) for t in time_name_ts.split("-")]
                if time_y_s < 1995:
                    ent_2_emb[h - ent_1_num, 0] += 1
                    time_y_s, time_m_s = 1995, 0
                if time_name_te != "~":
                    time_y_e, time_m_e = [int(t) for t in time_name_te.split("-")]
                    if time_y_e >= 1995:
                        ent_2_emb[h - ent_1_num, rel_time_cal(time_y_s, time_m_s):rel_time_cal(time_y_e, time_m_e)] += 1
        return np.array(ent_1_emb.tolist() + ent_2_emb.tolist())

    def save_embedding(self, save_dir:str="embeddings"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ### name
        if self.name_embeddings is not None:
            np.save(os.path.join(save_dir, "name_embeddings.npy"), self.name_embeddings)
        ### structure
        if self.struct_embeddings is not None:
            np.save(os.path.join(save_dir, "struct_embeddings.npy"), self.struct_embeddings)
        ### image
        if self.image_embeddings is not None:
            np.save(os.path.join(save_dir, "image_embeddings.npy"), self.image_embeddings)
        ### entity
        if self.entity_embeddings is not None:
            np.save(os.path.join(save_dir, "entity_embeddings.npy"), self.entity_embeddings)
