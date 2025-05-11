from math import floor
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
import scipy.stats as stats
class ScoreRange:
    def __init__(self):
        self.min = np.inf
        self.max = -np.inf
    def update(self, score):
        self.min = min(self.min, score)
        self.max = max(self.max, score)
    def normalize(self, score, upper=1.0):
        return (score - self.min) / (self.max - self.min) * upper
    def denormalize(self, score, upper=1.0):
        return self.min + (self.max - self.min) * score / upper
    def info(self):
        return self.min, self.max
    
class FineFSDataset(Dataset):
    def __init__(self, ann_root, feature_root, indices, config, fps=30, loader_normalize=True, type='tes'):
        self.type = type
        assert self.type in ['tes', 'pcs']
        self.indices = indices
        self.ann_files = sorted([f for f in os.listdir(ann_root) if f.endswith('.json')], key=lambda x: int(x.split('.')[0]))
        self.ann_root = ann_root
        self.feature_root = feature_root
        self.fps = fps
        self.action_name = ['nothing', 'jump', 'sequence', 'spin', 'unknown']
        self.action_dict = {self.action_name[i]:i for i in range(len(self.action_name))}
        self.element_list, self.element_dict = self.get_elements()
        self.score_range = ScoreRange()
        self.bv_range = ScoreRange()
        self.pcs_range = ScoreRange()
        self.tes_range = ScoreRange()
        self.pcs_total_range = ScoreRange()
        self.sop_range = ScoreRange()
        self.loader_normalize = loader_normalize
        self.config = config
        print('reading video info')
        self.read_video_info()
        self.version = 1.1
        
        print('score_range: ', self.score_range.info())
        print('bv_range: ', self.bv_range.info())
        print('pcs_range: ', self.pcs_range.info())
        print('tes_range: ', self.tes_range.info())

    def read_video_info(self):
        suffix_list = ['.mp4.npy', '.pkl']
        self.suffix = None
        for suffix in suffix_list:
            example_file = os.path.join(self.feature_root, '0' + suffix)
            if os.path.exists(example_file):
                self.suffix = suffix
                break
        if self.suffix is None:
            raise ValueError('No valid suffix found in the feature root')
        for ann_file in tqdm(self.ann_files):
            ann_path = os.path.join(self.ann_root, ann_file)
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            pcs_total = 0
            for k, v in ann['program_component'].items():
                self.pcs_range.update(v['score_of_pannel'])
                pcs_total += v['score_of_pannel']
            self.pcs_total_range.update(pcs_total)

            executed_element = ann['executed_element']
            tes = 0
            for element in executed_element.values():
                judge_score_list = element['judge_score']
                judge_score_list = sorted(judge_score_list)[1:len(judge_score_list)-1]
                score = sum(judge_score_list) / len(judge_score_list)
                self.score_range.update(score)
                bv = element['bv']
                self.bv_range.update(bv)
                tes += score * 0.1 * bv + bv
                sop = element['score_of_pannel']
                self.sop_range.update(sop)
            self.tes_range.update(tes)
            

    def get_elements(self):
        ann_list = os.listdir(self.ann_root)
        action = dict()
        for ann_file in ann_list:
            ann_path = os.path.join(self.ann_root, ann_file)
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            executed_element = ann['executed_element']
            for elements in executed_element.values():
                element = elements['element']
                if element in action:
                    action[element] += 1
                else:
                    action[element] = 1
        sorted_action = dict(sorted(action.items(), key=lambda item: item[1], reverse=True))
        element_list = list(sorted_action.keys())
        element_dict = {element_list[i]:i for i in range(len(element_list))}
        return element_list, element_dict
    
    def get_element_name(self):
        return self.element_list

    def get_element_dict(self):
        return self.element_dict
    
    def get_action_name(self):
        return self.action_name

    def get_action_dict(self):
        return self.action_dict

    def __len__(self):
        return len(self.indices)

    def get_soft_label(self, score, range, usdl_output_dim, usdl_std):
        soft_label = stats.norm.pdf(np.arange(usdl_output_dim), loc=(score - range.min) * (usdl_output_dim-1) / (range.max - range.min), scale=usdl_std).astype(np.float32)
        soft_label = soft_label / soft_label.sum()
        soft_label = torch.from_numpy(soft_label)
        return soft_label

    def normalize(self, score, range):
        return (score - range.min) / (range.max - range.min)
    
    def __getitem__(self, idx):
        ann_name = self.ann_files[self.indices[idx]]
        feature_name = ann_name[:-5] + self.suffix
        with open(os.path.join(self.ann_root, ann_name), 'r') as f:
            data = json.load(f)
        if self.suffix == '.mp4.npy':
            feature = np.load(os.path.join(self.feature_root, feature_name))
        else:
            feature = torch.load(os.path.join(self.feature_root, feature_name))
        vlen = feature.shape[0]
        action_list = np.zeros(vlen, dtype=int)
        
        
        executed_element = data['executed_element']
        actions_info = np.zeros((len(executed_element), 4), dtype=int)
        actions_score = np.zeros((len(executed_element), 2), dtype=float) # score, bv
        sop_list = np.zeros(len(executed_element), dtype=float)

        # pcs
        program_component = data['program_component']
        pcs_score = []
        pcs_soft_label = []
        for k, v in program_component.items():
            pcs_score.append(v['score_of_pannel'])
            pcs_soft_label.append(self.get_soft_label(v['score_of_pannel'], self.pcs_range, self.config.usdl_output_dim, self.config.usdl_std))
        pcs_total = sum(pcs_score)

        score_soft_label = []
        bv_soft_label = []

        attn_mask = torch.zeros(len(executed_element), vlen, dtype=torch.int64)

        for i, element in enumerate(executed_element.values()):
            # get coarse_class
            coarse_class = element.get('coarse_class', 'unknown')
            if coarse_class is None:
                coarse_class = 'unknown'
                
            action_time = element['time']
            start, end = action_time.split(',')
            start = floor((int(start.split('-')[0])*60 + int(start.split('-')[1])) * self.fps)
            end = floor((int(end.split('-')[0])*60 + int(end.split('-')[1])) * self.fps)
            
            action_index = self.action_dict[coarse_class]
            action_list[start:end] = action_index
            # get judge score
            judge_score_list = element['judge_score']
            judge_score_list = sorted(judge_score_list)[1:len(judge_score_list)-1]
            score = sum(judge_score_list) / len(judge_score_list)
            # get element
            element_name = element['element']
            element_index = self.element_dict[element_name]
            # get bv
            bv = element['bv']
            actions_info[i, :] = np.array([start, end, element_index, action_index])
            # get sop
            sop = element['score_of_pannel']
            sop_list[i] = sop
            if self.loader_normalize:
                actions_score[i, :] = np.array([self.score_range.normalize(score), self.bv_range.normalize(bv)])
            else:
                actions_score[i, :] = np.array([score, bv])
            
            end = min(end, vlen - 1)
            start = min(start, end, vlen - 1)
            
            # usdl soft label
            score_soft_label.append(self.get_soft_label(score, self.score_range, self.config.usdl_output_dim, self.config.usdl_std))
            bv_soft_label.append(self.get_soft_label(bv, self.bv_range, self.config.usdl_output_dim, self.config.usdl_std))
            
            # attn mask
            attn_mask[i, start:end] = 1
            
            
        tes_total = sum(sop_list)
        mask = torch.ones(len(self.action_dict.keys()), vlen, dtype=torch.int64)
        tes_total = self.normalize(tes_total, self.tes_range)
        pcs_total = self.normalize(pcs_total, self.pcs_total_range)
        ret_score = tes_total if self.type == 'tes' else pcs_total
        return {
            'feature': torch.FloatTensor(feature).transpose(1, 0),
            'score': ret_score,
            
        }

def collate_fn(batch):
    """
    对长度不一致的数据进行padding。
    """
    # 解包字典列表
    features = [item['feature'] for item in batch]
    scores = [item['score'] for item in batch]

    # features
    max_len = max(tensor.size(1) for tensor in features)
    padded_features = pad_sequence([torch.nn.functional.pad(tensor, (0, max_len - tensor.size(1)), value=0) for tensor in features], batch_first=True)
    padded_features = padded_features.permute(0, 2, 1)
    
    scores = torch.tensor(scores, dtype=torch.float32)
    return padded_features, scores

def get_FineFS_loader(ann_root, feature_root, indices_path, batch_size, num_workers=4, fps=30, loader_normalize=True, type='tes'):
    with open(indices_path, 'rb') as f:
        indices = pickle.load(f)
    train_indices = indices['train']
    test_indices = indices['test']
    from types import SimpleNamespace
    config = SimpleNamespace(usdl_output_dim=101, usdl_std=10)
    train_dataset = FineFSDataset(ann_root, feature_root, train_indices, config, fps, loader_normalize, type)
    test_dataset = FineFSDataset(ann_root, feature_root, test_indices, config, fps, loader_normalize, type)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    return train_dataset, test_dataset, train_dataloader, test_dataloader

if __name__ == '__main__':
    ann_root = 'FineFS/data/annotation'
    feature_root = 'FineFS/data/video_features'
    indices_path = 'indices/indices_fs_0.pkl'
    
    train_dataset, test_dataset, train_dataloader, test_dataloader = get_FineFS_loader(ann_root, feature_root, indices_path, 2, loader_normalize=False, type='pcs')
    for batch in train_dataloader:
        features, scores = batch
        # print(features.shape)
        print(scores)
        # break