import vllm 
import torch 
from torch import Tensor 


class VLLMEmbeddingModel:
    def __init__(
        self,
        checkpoint: str,
    ):
        self.model = vllm.LLM(model=checkpoint, task="embed")

    def forward(
        self,
        text_list: list[str],
        query_instruct: str = '',
        embedding_dim: int = -1,
    ) -> Tensor:
        batch_size = len(text_list)

        if query_instruct:
            text_list = [
                f'Instruct: {query_instruct}\nQuery:{text}'
                for text in text_list 
            ]

        output_list = self.model.embed(text_list)
        embedding_2d = torch.tensor([o.outputs.embedding for o in output_list], dtype=torch.float32) 

        if embedding_dim > 0:
            embedding_2d = embedding_2d[:, :embedding_dim]
        else:
            embedding_dim = embedding_2d.shape[-1]
        assert embedding_2d.shape == (batch_size, embedding_dim) 

        return embedding_2d 
