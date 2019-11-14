import torch
import torch.nn as nn
import torch.nn.functional as F
from common.cmd_args import cmd_args


class EmbeddingTable(nn.Module):
    def __init__(self, num_ents, latent_dim):
        super(EmbeddingTable, self).__init__()

        self.num_ents = num_ents
        self.ent_embeds = nn.Embedding(self.num_ents, latent_dim)

    def forward(self, batch_data):
        """

        :param batch_data:
            tensor-like index of size (b)
        :return:
            embeddings of size (b, latent_dim)

        """

        node_embeds = self.ent_embeds(torch.tensor(batch_data, dtype=torch.long, device=cmd_args.device))
        return node_embeds