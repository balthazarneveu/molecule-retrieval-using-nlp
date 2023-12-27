
from transformers import AutoModel
from generic import GenericModel


class TextEncoder(GenericModel):
    def __init__(self, model_name: str = 'distilbert-base-uncased', freeze: bool = True):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:, 0, :]


if __name__ == '__main__':
    text_encoder = TextEncoder()
    print(text_encoder.count_params())
