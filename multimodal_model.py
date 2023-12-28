from generic import GenericModel
import time


class MultimodalModel(GenericModel):
    def __init__(self, graph_encoder: GenericModel, text_encoder: GenericModel):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder

    def forward(self, graph_batch, input_ids, attention_mask):
        # Time profiling for graph_encoder
        start_time = time.time()
        graph_encoded = self.graph_encoder(graph_batch)
        graph_time = time.time() - start_time
        print(f"Graph encoding time: {graph_time:.4f} seconds")

        # Time profiling for text_encoder
        start_time = time.time()
        print(input_ids.shape)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        text_time = time.time() - start_time
        print(f"Text encoding time: {text_time:.4f} seconds")
        return graph_encoded, text_encoded

    def count_params(self):
        print(f"Graph : {self.graph_encoder.count_params()*1E-3:.1f}k")
        print(f"Text: {self.text_encoder.count_params()*1E-6:.1f}M")
        return super().count_params()


if __name__ == "__main__":
    from language_model import TextEncoder
    from graph_model import BasicGraphEncoder
    graph_encoder = BasicGraphEncoder(num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300)
    text_encoder = TextEncoder(freeze=True)
    model = MultimodalModel(graph_encoder, text_encoder)
    print(model.count_params())
