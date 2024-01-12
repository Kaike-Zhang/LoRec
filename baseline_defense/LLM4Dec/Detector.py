import torch
import torch.nn as nn


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

    def forward(self, x):
        x = self.active(self.drop(self.bn1(self.linear(x))))
        return self.bn2(self.proj(x))


class Detector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                item_embedding_dim=config["hidden_units"],
                                layers=[config["hidden_units"]] * (config["dnn_layer"] + 1),
                                drop_rate=config["dropout_rate"])
        
        self.proj = ProjLayer(in_dim=config["hidden_units"], hidden_dim=128, drop_rate=config["dropout_rate"])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        score = self.proj(x)

        return self.sigmoid(score)
