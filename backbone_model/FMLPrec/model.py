import numpy as np
import torch
import torch.nn as nn
from backbone_model.FMLPrec.modules import Encoder, LayerNorm, MLP_Layers

class BasicFMLPrec(nn.Module):
    def __init__(self, config):
        super(BasicFMLPrec, self).__init__()
        self.config = config
        self.item_embeddings = nn.Embedding(config["n_items"]+1, self.config["hidden_units"], padding_idx=0)
        self.max_seq_len = config["maxlen"] + 1
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.config["hidden_units"])
        self.LayerNorm = LayerNorm(self.config["hidden_units"], eps=1e-12)
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.item_encoder = Encoder(config)

        self.apply(self.init_weights)

    # same as SASRec
    def forward(self, interaction_list, interaction_mask, neg_list):

        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])[:, :-1]
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        input_logs_embs = input_embs_all[:, :-1, :]
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.item_emb(torch.LongTensor(np.array(neg_list)).to(self.config["device"]))[:, :-1, :]

        
        seq_length = input_embs_all.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs_all.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_embs_all)
        input_logs_embs = input_logs_embs + self.position_embeddings(position_ids)

        input_logs_embs = self.LayerNorm(input_logs_embs)
        input_logs_embs = self.dropout(input_logs_embs)

        prec_vec = self.item_encoder(input_logs_embs,
                                                log_mask,
                                                output_all_encoded_layers=True,
                                                )[-1]

        return prec_vec, target_pos_embs, target_neg_embs, prec_vec
    

    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        item_embs = self.item_emb(torch.LongTensor(np.array(item_indices)).to(self.config["device"]))

        seq_length = input_logs_embs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_logs_embs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_logs_embs)
        input_logs_embs = input_logs_embs + self.position_embeddings(position_ids)

        input_logs_embs = self.LayerNorm(input_logs_embs)
        input_logs_embs = self.dropout(input_logs_embs)

        prec_vec = self.item_encoder(input_logs_embs,
                                                log_mask,
                                                output_all_encoded_layers=True,
                                                )[-1][:, -1, :]
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1)

        return logits


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config["initializer_range"])
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




class LLMFMLPrec(nn.Module):
    def __init__(self, config):
        super(LLMFMLPrec, self).__init__()
        self.config = config

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                 item_embedding_dim=config["hidden_units"],
                                 layers=[config["hidden_units"]] * (config["dnn_layer"] + 1),
                                 drop_rate=config["dropout_rate"])

        self.max_seq_len = config["maxlen"] + 1
        self.position_embeddings = nn.Embedding(self.max_seq_len, self.config["hidden_units"])
        self.LayerNorm = LayerNorm(self.config["hidden_units"], eps=1e-12)
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.item_encoder = Encoder(config)

        self.apply(self.init_weights)

    # same as SASRec
    def forward(self, interaction_list, interaction_mask, neg_list):

        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])[:, :-1]
        input_embs_all = self.fc(interaction_list)
        input_logs_embs = input_embs_all[:, :-1, :]
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.fc(neg_list)[:, :-1, :]

        seq_length = log_mask.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        input_logs_embs = input_logs_embs + self.position_embeddings(position_ids)

        input_logs_embs = self.LayerNorm(input_logs_embs)
        input_logs_embs = self.dropout(input_logs_embs)

        prec_vec = self.item_encoder(input_logs_embs,
                                                log_mask,
                                                output_all_encoded_layers=True,
                                                )[-1]
        
        with torch.no_grad():
            combine_emb = self.get_emb(interaction_list, interaction_mask)

        return prec_vec, target_pos_embs, target_neg_embs, combine_emb
    

    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.fc(interaction_list)
        item_embs = self.fc(item_indices)

        seq_length = log_mask.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        input_logs_embs = input_logs_embs + self.position_embeddings(position_ids)

        input_logs_embs = self.LayerNorm(input_logs_embs)
        input_logs_embs = self.dropout(input_logs_embs)

        prec_vec = self.item_encoder(input_logs_embs,
                                                log_mask,
                                                output_all_encoded_layers=True,
                                                )[-1][:, -1, :]
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        input_logs_embs = input_embs_all

        seq_length = log_mask.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        input_logs_embs = input_logs_embs + self.position_embeddings(position_ids)
        
        input_logs_embs = self.LayerNorm(input_logs_embs)
        input_logs_embs = self.dropout(input_logs_embs)

        prec_vec = self.item_encoder(input_logs_embs,
                                                log_mask,
                                                output_all_encoded_layers=True,
                                                )[-1]

        return torch.cat((input_embs_all, prec_vec), dim=2)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config["initializer_range"])
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
