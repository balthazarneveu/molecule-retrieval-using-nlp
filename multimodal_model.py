from generic import GenericModel
import torch


class MultimodalModel(GenericModel):
    def __init__(self, graph_encoder: GenericModel, text_encoder: GenericModel, temperature: bool = False):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
        if temperature:
            self.temperature = torch.nn.Parameter(temperature*torch.ones(1))

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def count_params(self):
        print(f"Graph : {self.graph_encoder.count_params()*1E-3:.1f}k")
        print(f"Text: {self.text_encoder.count_params()*1E-6:.1f}M")
        return super().count_params()

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder


if __name__ == "__main__":
    from language_model import TextEncoder
    from graph_model import BasicGraphEncoder
    graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    text_encoder = TextEncoder(freeze=True)
    model = MultimodalModel(graph_encoder, text_encoder)
    print(model.count_params())
