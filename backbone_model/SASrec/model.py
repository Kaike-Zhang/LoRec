import numpy as np
import torch
from torch import nn
from backbone_model.SASrec.encoders import User_Encoder, MLP_Layers
from torch.nn.init import xavier_normal_, constant_


class LLMSASrec(torch.nn.Module):
    def __init__(self, config):
        super(LLMSASrec, self).__init__()
        self.config = config
        self.max_seq_len = config["maxlen"] + 1

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                 item_embedding_dim=config["hidden_units"],
                                 layers=[config["hidden_units"]] * (config["dnn_layer"] + 1),
                                 drop_rate=config["dropout_rate"])

        self.user_encoder = User_Encoder(
            item_num=config["n_items"],
            max_seq_len=config["maxlen"],
            item_dim=config["hidden_units"],
            num_attention_heads=config["num_heads"],
            dropout=config["dropout_rate"],
            n_layers=config["num_blocks"])


    def forward(self, interaction_list, interaction_mask, neg_list):
        # log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])[:, :-1]
        # input_embs_all = self.fc(interaction_list)
        # input_logs_embs = input_embs_all[:, :-1, :]
        # target_pos_embs = input_embs_all[:, 1:, :]
        # target_neg_embs = self.fc(neg_list)[:, :-1, :]

        # prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        # return prec_vec, target_pos_embs, target_neg_embs
        # pos_score = (prec_vec * target_pos_embs).sum(-1)
        # neg_score = (prec_vec * target_neg_embs).sum(-1)

        # return pos_score, neg_score
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        input_logs_embs = input_embs_all
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.fc(neg_list)[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return prec_vec[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, prec_vec), dim=2)
    
    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.fc(interaction_list)
        item_embs = self.fc(item_indices)

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])[:, -1, :]
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1)

        return logits
    
    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        input_logs_embs = input_embs_all

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return torch.cat((input_embs_all, prec_vec), dim=2)
    

class BasicSASrec(torch.nn.Module):
    def __init__(self, config):
        super(BasicSASrec, self).__init__()
        self.config = config
        self.max_seq_len = config["maxlen"] + 1
        self.item_emb = torch.nn.Embedding(config["n_items"]+1, self.config["hidden_units"], padding_idx=0)
        xavier_normal_(self.item_emb.weight.data[1:])

        self.user_encoder = User_Encoder(
            item_num=config["n_items"],
            max_seq_len=config["maxlen"],
            item_dim=config["hidden_units"],
            num_attention_heads=config["num_heads"],
            dropout=config["dropout_rate"],
            n_layers=config["num_blocks"])


    def forward(self, interaction_list, interaction_mask, neg_list):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        input_logs_embs = input_embs_all
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.item_emb(torch.LongTensor(np.array(neg_list)).to(self.config["device"]))[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return prec_vec[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, prec_vec), dim=2)
        # return prec_vec, target_pos_embs, target_neg_embs, prec_vec
        # pos_score = (prec_vec * target_pos_embs).sum(-1)
        # neg_score = (prec_vec * target_neg_embs).sum(-1)

        # return pos_score, neg_score
    
    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        item_embs = self.item_emb(torch.LongTensor(np.array(item_indices)).to(self.config["device"]))

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])[:, -1, :]
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        input_logs_embs = input_embs_all

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return torch.cat((input_embs_all, prec_vec), dim=2)