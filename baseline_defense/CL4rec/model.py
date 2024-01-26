import numpy as np
import torch
from torch import nn
from baseline_models.CL4rec.modules import compute_info_nce_loss
from baseline_models.CL4rec.encoders import User_Encoder, MLP_Layers
from torch.nn.init import xavier_normal_, constant_


class LLMCL4rec(torch.nn.Module):
    def __init__(self, config):
        super(LLMCL4rec, self).__init__()
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
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        input_logs_embs = input_embs_all
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.fc(neg_list)[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return prec_vec[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, prec_vec), dim=2)
    

    def crop_forward_loss(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)

        non_zero_counts = log_mask.sum(dim=1)
        L_prime = int(non_zero_counts.max().item() * 0.5) 

        min_start_pos = int(log_mask.size(1) - non_zero_counts.max().item())

        start_pos = torch.randint(min_start_pos, log_mask.size(1) - L_prime, (1,)).item()

        input_embs_1 = input_embs_all[:, start_pos:start_pos + L_prime, :]
        input_embs_2 = input_embs_all[:, start_pos:start_pos + L_prime, :]

        mask_1 = log_mask[:, start_pos:start_pos + L_prime]
        mask_2 = log_mask[:, start_pos:start_pos + L_prime]


        w = torch.ones((log_mask.size(0), 1), dtype=torch.float32).to(self.config["device"])
        w[(mask_1.sum(dim=1) == 0) | (mask_2.sum(dim=1) == 0)] = 0

        # prediction vector BxD
        prec_vec_1 = self.user_encoder(input_embs_1, mask_1, self.config["device"])[:, -1, :]
        prec_vec_2 = self.user_encoder(input_embs_2, mask_2, self.config["device"])[:, -1, :]


        filtered_prec_vec_1 = prec_vec_1[w.squeeze() == 1]
        filtered_prec_vec_2 = prec_vec_2[w.squeeze() == 1]

        info_nce_loss = compute_info_nce_loss(filtered_prec_vec_1, filtered_prec_vec_2)

        return info_nce_loss

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
    

class BasicCL4rec(torch.nn.Module):
    def __init__(self, config):
        super(BasicCL4rec, self).__init__()
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
    

    def crop_forward_loss(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"]) # B x L
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"])) # B x L x D

        non_zero_counts = log_mask.sum(dim=1)
        L_prime = int(non_zero_counts.max().item() * 0.5) 

        min_start_pos = log_mask.size(1) - non_zero_counts.max().item()

        start_pos = torch.randint(min_start_pos, log_mask.size(1) - L_prime, (1,)).item()

        input_embs_1 = input_embs_all[:, start_pos:start_pos + L_prime, :]
        input_embs_2 = input_embs_all[:, start_pos:start_pos + L_prime, :]

        mask_1 = log_mask[:, start_pos:start_pos + L_prime]
        mask_2 = log_mask[:, start_pos:start_pos + L_prime]


        w = torch.ones((log_mask.size(0), 1), dtype=torch.float32).to(self.config["device"])
        w[(mask_1.sum(dim=1) == 0) | (mask_2.sum(dim=1) == 0)] = 0

        # prediction vector BxD
        prec_vec_1 = self.user_encoder(input_embs_1, mask_1, self.config["device"])[:, -1, :]
        prec_vec_2 = self.user_encoder(input_embs_2, mask_2, self.config["device"])[:, -1, :]


        filtered_prec_vec_1 = prec_vec_1[w.squeeze() == 1]
        filtered_prec_vec_2 = prec_vec_2[w.squeeze() == 1]

        info_nce_loss = compute_info_nce_loss(filtered_prec_vec_1, filtered_prec_vec_2)

        return info_nce_loss
        
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
