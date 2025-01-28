import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Union, Optional
import random
import yaml
import numpy as np
import pandas as pd
import torch
import json
import sys
import os

from datasets import load_dataset, load_from_disk
from git import Repo


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class MDSUPDRS(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        
        self.config = cfg # .datasets

        self.labels, self.prompts = self._load_data(cfg.label_path, cfg.prompt_path)
        
        self.prefix = self._get_prefix()
        self.rec_ids = [(key, subkey) for key, subdict in self.prompts.items() for subkey in subdict.keys()] # if key in self.prompts]
        self.section = cfg.section
        self.event_id = cfg.event_id
        # print(f"Number of records with prompt: {len(self.rec_ids)}")
        
        self.debug = self.config.get('debug', False)  # Debug flag to limit the dataset to n rows
        n = 4
        if self.debug:
            print(f"Debug mode enabled. Limiting dataset to the first {n} records.")
            self.rec_ids = self.rec_ids[:n+20]
            
        self.prompts = self._get_prompts(section=self.section)
        self.rec_ids = self.rec_ids[cfg.start_row_num:]
        if self.debug:
            self.rec_ids = self.rec_ids[:n]
        
    def _load_data(self, label_path, prompt_path):
        
        with open (label_path, 'r') as f:
            label_dict = json.load(f)
            
        with open (prompt_path, 'r') as f:
            prompts = json.load(f)

        return label_dict, prompts
        
    def _get_prefix(self):
        
        prefix = '''
        The MDS-Unified Parkinson's Disease Rating Scale (MDS-UPDRS) was developed to evaluate various aspects of Parkinson’s disease including non-motor and motor experiences of daily living and motor complications. It includes a motor evaluation and characterizes the extent and burden of disease across various populations.
        Below are the responses of a patient to the MDS-UPDRS questionnaire. Each question is followed by instructions for the patient, examiner, and/or caregiver, as well as the final response.
        '''

        prefix = prefix.replace('\t', '').replace('    ', '').strip()
        return prefix
    
    def _get_prompts(self, section='all'):
        prompts_result = {}
        separator = "\n\n" + "- - - " * 50 + "\n\n"
        
        rec_ids_to_remove = []
        event_ids = ['BL', 'V02', 'V04', 'V06', 'V08'] if self.event_id=="all" else [self.event_id]

        for i in range(len(self.rec_ids)):
            patient_id, event_id = self.rec_ids[i]
            
            if event_id not in event_ids:
                rec_ids_to_remove.append(self.rec_ids[i])
                continue
            
            try:
                prompt = self.prompts[patient_id][event_id]
            except KeyError:
                print(f"Error: No prompt found for {patient_id} {event_id}")
                rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                continue  # Skip to the next record
            
            if section == 'all':
                prompt = "\n\n".join(prompt)
            else:
                if str(section) == '1':
                    # import pdb; pdb.set_trace()
                    prompt = [p.strip() for p in prompt if '1.' in p.strip()[:3]]
                    if not prompt:
                        # print(f"No section '1' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue
                elif str(section) == '2':
                    prompt = [p.strip() for p in prompt if '2.' in p.strip()[:3]]
                    if not prompt:
                        # print(f"No section '2' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue
                elif str(section) == '3':
                    prompt = [p.strip() for p in prompt if '3.1' in p.strip()[:5]]
                    if not prompt:
                        # print(f"No section '3.1' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue
                elif str(section) == '4':
                    prompt = [p.strip() for p in prompt if '4.' in p.strip()[:3]]
                    if not prompt:
                        # print(f"No section '4' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue

                elif str(section) == '23':
                    prompt_sec2 = [p.strip() for p in prompt if '2.' in p.strip()[:3]] 
                    prompt_sec3 = [p.strip() for p in prompt if '3.' in p.strip()[:3]]
                    # import pdb; pdb.set_trace()
                    if not prompt_sec2:
                        # print(f"No section '2' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue
                    if not prompt_sec3:
                        # print(f"No section '3' found for {patient_id} {event_id}")
                        rec_ids_to_remove.append(self.rec_ids[i])  # Mark for removal
                        continue
                    prompt = prompt_sec2 + prompt_sec3
                        
                prompt = "\n\n".join(prompt)  
            
            formatted_prompt = self.prefix + separator + prompt
            
            if patient_id not in prompts_result:
                prompts_result[patient_id] = {}
            prompts_result[patient_id][event_id] = formatted_prompt
        
        # Remove invalid entries from self.rec_ids
        for rec in rec_ids_to_remove:
            self.rec_ids.remove(rec)
        
        return prompts_result
    
    def __getitem__(self, idx):
        """Retrieve an item by index."""
        
        patient_id, event_id = self.rec_ids[idx][0], self.rec_ids[idx][1]
        prompt = self.prompts[patient_id][event_id]
        
        try:
            label = self.labels[patient_id][event_id]
        except:
            # print(f"Error: No label found for {patient_id} {event_id}")
            label = None

        return {
            'patient_id': patient_id,
            'event_id': event_id,
            'input_prompt': prompt,
            'label': label,
        }
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.rec_ids)
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle batching.
        This function takes a list of individual items (returned by __getitem__)
        and combines them into a batch.
        """
    
        patient_id_batch = [item['patient_id'] for item in batch]
        event_id_batch = [item['event_id'] for item in batch]
        input_prompt_batch = [item['input_prompt'] for item in batch]
        label_batch = [item['label'] for item in batch]
        
        return {
            'patient_id': patient_id_batch,
            'event_id': event_id_batch,
            'input_prompt': input_prompt_batch,
            'label': label_batch
        }
        
        