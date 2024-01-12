import torch
import torch.nn as nn
from models.Transformer_Block.position_emb import PositionEmb
from models.Transformer_Block.transformer_block import TransformerBlock
import torch.nn.functional as F


class MLP_Layers(nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def forward(self, sample_items):
        sample_items = self.activate(self.fc(sample_items))
        return self.mlp_layers(sample_items)
    

class ProjLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=drop_rate)
        self.active = nn.GELU()
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            # if module.bias is not None:
            #     nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.active(self.drop(self.bn1(self.linear(x))))
        return self.bn2(self.proj(x))


class LCTer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                item_embedding_dim=2 * config["hidden_units"],
                                layers=[2 * config["hidden_units"]] * (config["dnn_layer"] + 1),
                                drop_rate=config["dropout_rate"])

        self.transformer_layers = nn.ModuleList([TransformerBlock(2 * config["hidden_units"], config["heads"], config["ffd_hidden"], config["dropout_rate"])
                                          for _ in range(config.get("block_num", 1))])
        
        self.seq_emb = torch.nn.Embedding(1, 2 * self.config["hidden_units"])
        nn.init.xavier_normal_(self.seq_emb.weight.data)

        self.proj = ProjLayer(in_dim=2 * config["hidden_units"], hidden_dim=128, drop_rate=config["dropout_rate"])

        self.sigmoid = nn.Sigmoid()
        
        self.position_ecd = PositionEmb(2 * config["hidden_units"], config.get("max_len", 50))

    def forward(self, seq, detections=None, mask=None):
        if detections is not None:
            detections = self.fc(detections)
        seq = seq + self.position_ecd(seq)
        seq_embs = self.seq_emb(torch.tensor([0]).to(seq.device))
        seq_embs = seq_embs.unsqueeze(1).expand(seq.size(0), -1, -1)
        x = torch.cat((seq_embs, seq), dim=1)
    

        if mask is not None:
            ones_tensor = torch.ones(seq_embs.size(0), 1, seq_embs.size(2), dtype=torch.bool).to(seq.device)
            mask = torch.FloatTensor(mask).unsqueeze(-1).expand_as(seq).to(seq.device)
            mask = torch.cat((ones_tensor, mask), dim=1)
        

        for layer in self.transformer_layers:
            if mask is not None:
                x *= mask
            x = layer(x)
        
        x = x[:,0,:].squeeze()
        if detections is not None:
            similarity = F.cosine_similarity(x, detections, dim=1).mean()
        else:
            similarity = None

        score = self.proj(x)

        return self.sigmoid(score), similarity
