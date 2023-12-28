
from transformers import AutoModel
from generic import GenericModel
import torch
import pickle
from pathlib import Path
from typing import Optional
import logging


class TextEncoder(GenericModel):
    def __init__(
        self, model_name: str = 'distilbert-base-uncased', freeze: bool = True,
            cache_dir: Optional[Path] = None):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        cache = False
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache = True
        else:
            if cache_dir is not None:
                logging.warning("Cache is not supported when training the language model")
        self.cache = cache
        self.cache_dir = cache_dir

    def forward(self, input_ids, attention_mask):
        encoded_text = self.load_cached_embeddings(input_ids)
        if encoded_text is None:
            encoded_text = self.bert(input_ids, attention_mask=attention_mask)
            self.cache_embeddings(input_ids, encoded_text)
        return encoded_text.last_hidden_state[:, 0, :]

    def get_input_hash(self, input_ids):
        # Simple hash function for tensor
        return torch.sum(input_ids * torch.arange(input_ids.numel(), device=input_ids.device)).item()

    def __get_cache_path(self, input_id):
        input_hash = self.get_input_hash(input_id)
        file_path = self.cache_dir/f"{input_hash}.pkl"
        return input_hash, file_path

    def cache_embeddings(self, input_ids, embeddings):
        _input_hash, file_path = self.__get_cache_path(input_ids)

        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)

    def load_cached_embeddings(self, input_ids):
        if self.cache is False:
            return None
        _input_hash, file_path = self.__get_cache_path(input_ids)

        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None


if __name__ == '__main__':
    text_encoder = TextEncoder()
    print(text_encoder.count_params())
