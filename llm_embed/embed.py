import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Union, Optional
import numpy as np
import os
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_manhattan_distances, paired_euclidean_distances
import logging
import hydra
from omegaconf import DictConfig
from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Optional, Sequence, Union
import openai
from tqdm import tqdm
from datetime import datetime

# time_string = datetime.now().strftime('%Y%m%d_%H%M%S')

torch.cuda.empty_cache()

class LLMEmbed:
    def __init__(
        self,
        dataset,  # Expecting an instance of a dataset containing single sentences
        batch_size: int = 32,
        limit: int | None = None,
        save_path: str | None = None,  # Path to save batch embeddings
        **kwargs,
    ):
        """
        STS evaluator for datasets with single sentences.

        Args:
            dataset: The dataset object containing single sentences and their associated scores.
            batch_size (int): Batch size for processing.
            limit (int): Optional limit to reduce the dataset size for testing purposes.
            save_path (str): Directory path to save embeddings after each batch iteration.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_path = './embeddings'
        
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
        )

    
    def save_embeddings(self, batch_index: int, model_configs, dataset_configs, tokenizer_configs, ids: List[str], event_ids: List[str], embeddings: List[torch.Tensor], input_tokens: List[List[int]], input_prompt: List[str], generated_text=None):
        """
        Saves embeddings and input tokens to file for each batch iteration.

        Args:
            batch_index (int): The current batch index.
            embeddings (torch.Tensor): Embeddings to save.
            input_tokens (List[List[int]]): Input tokens corresponding to each sentence in the batch.
        """
        
        datetime_str = datetime.now().strftime("%d%b%H%M%S").lower()
        
        if self.save_path:
            
            # Create the directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            model_name = model_configs.model_name.split('/')[-1]
            
            save_filename = f"{self.save_path}/batch_{batch_index}_data_{model_name}_{datetime_str}.pkl" # not used anymore
            if batch_index == -1:
                save_filename = f"{self.save_path}/{model_name}_section{dataset_configs.section}_{dataset_configs.event_id}_{datetime_str}.pkl"
            
            # Save data as a pickle file
            save_data = {
                "patient_ids": ids,
                "event_ids": event_ids,
                "embeddings": embeddings,
                "input_tokens": input_tokens,
                "input_prompt": input_prompt,
                "model_configs": model_configs, 
                "tokenizer_configs": tokenizer_configs, 
                # "labels": [self.dataset.labels[id] for id in ids if id in self.dataset.labels else None]
                "labels": [self.dataset.labels[id] if id in self.dataset.labels else None for id in ids]
            }
            
            with open(save_filename, "wb") as f:
                pickle.dump(save_data, f)
            
            print(f"Saved embeddings and input tokens for batch {batch_index} to {save_filename}")

    def __call__(
        self,
        model: Any,  # This can be HuggingFaceEncoder, OpenAIEncoder, etc.
    ):
        """
        Evaluate the model on the STS dataset with single sentences.

        Args:
            model (Any): The encoder model, could be HuggingFaceEncoder, OpenAIEncoder, etc.

        Returns:
            dict: Evaluation results with Pearson and Spearman correlations for various distance metrics.
        """

        accelerator = model.accelerator

        model_configs, tokenizer_configs = model.model_configs, model.tokenizer_configs
        dataset_configs = model.dataset_configs
        
        save_data = {'batch_index': None, 'patient_ids': [], 'event_ids': [], 'input_tokens': [], 'input_prompt': [], 'embeddings': []}

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Processing Batches")):
            
            sentences = batch['input_prompt'] # batch['sentence']
            input_tokens, mean_pooled, last_token = model.encode(sentences)

            for i in range(len(sentences)):

                embeddings_mean_pooled, embeddings_last_token = mean_pooled[i], last_token[i]
                
                save_data['patient_ids'].append(batch['patient_id'][i])
                save_data['input_tokens'].append(input_tokens[i])
                save_data['input_prompt'].append(sentences[i])
                # save_data['embeddings'].append(embeddings_subsampled)
                save_data['embeddings'].append({
                    "mean_pooled": embeddings_mean_pooled,
                    "last_token": embeddings_last_token,
                })   
                
                save_data['event_ids'].append(batch['event_id'][i])

            # self.save_embeddings(batch_idx, model_configs, tokenizer_configs, ids=save_data['patient_ids'], event_ids=save_data['event_ids'], embeddings=save_data['embeddings'], input_tokens=save_data['input_tokens'], input_prompt=save_data['input_prompt'])
        if accelerator.is_main_process:
            self.save_embeddings(-1, model_configs, dataset_configs, tokenizer_configs, ids=save_data['patient_ids'], event_ids=save_data['event_ids'], embeddings=save_data['embeddings'], input_tokens=save_data['input_tokens'], input_prompt=save_data['input_prompt'])
        
        return True
    
    


    

