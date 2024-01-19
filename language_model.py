
from transformers import AutoModel, AutoModelForCausalLM
from generic import GenericModel
import torch
from typing import Optional
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig


class TextEncoder(GenericModel):
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        freeze: bool = True,
        lora: Optional[dict] = None,
        adapter: Optional[GenericModel] = None,
        quantization_config=None
    ):
        super(TextEncoder, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name, quantization_config=quantization_config)
        if lora is not None and lora != {}:
            peft_config = LoraConfig(**lora)
            self.bert = get_peft_model(self.bert, peft_config)
            self.bert.print_trainable_parameters()

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.adapter = adapter

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        out = encoded_text.last_hidden_state[:, 0, :]
        if self.adapter is not None:
            out = self.adapter(out)
        return out


class Adapter(GenericModel):
    def __init__(self, text_embedding_size=768, hdim=512, out_size=512):
        super().__init__()
        self.fc1 = torch.nn.Linear(text_embedding_size, hdim)
        self.fc2 = torch.nn.Linear(hdim, out_size)
        self.relu = torch.nn.ReLU()
        self.adapter = torch.nn.Sequential(self.fc1, self.relu, self.fc2)

    def forward(self, x):
        return self.adapter(x)


if __name__ == '__main__':
    adapt = Adapter()
    # adapt = None
    text_encoder = TextEncoder(adapter=adapt)
    print(text_encoder.count_params())
