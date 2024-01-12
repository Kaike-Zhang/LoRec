import numpy as np
from torch import nn
import torch
from torch.nn.init import xavier_normal_, constant_

class BasicGRU4rec(nn.Module):
    def __init__(self, config, final_act='tanh'):
        super(BasicGRU4rec, self).__init__()
        self.config = config
        self.input_size = self.config["n_items"] + 1
        self.hidden_size = self.config["hidden_units"]
        self.output_size = self.config["hidden_units"]
        self.num_layers = self.config["num_blocks"]
        self.dropout_hidden = self.config["dropout_rate"]
        self.dropout_input = self.config["dropout_rate"]
        self.embedding_dim = self.config["hidden_units"]
        self.batch_size = self.config["batch_size"]
        self.device = self.config["device"]
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        self.create_final_activation(final_act)
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(self.input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward_step(self, input, hidden):
        output, hidden = self.gru(input, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def forward(self, interaction_list, interaction_mask, neg_list):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.look_up(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.item_emb(torch.LongTensor(np.array(neg_list)).to(self.config["device"]))[:, :-1, :]
        hidden = self.init_hidden(input_embs_all.size(0))
        final_emb_list = []
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            final_emb_list.append(output.squeeze())
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        
        final_output = torch.stack(final_emb_list).transpose(0, 1)

        return final_output[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, final_output), dim=2)


    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.look_up(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        item_embs = self.look_up(torch.LongTensor(np.array(item_indices)).to(self.config["device"]))
        hidden = self.init_hidden(input_embs_all.size(0))
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        logits = item_embs.matmul(output.unsqueeze(-1)).squeeze(-1)

        return logits
    
    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.look_up(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        hidden = self.init_hidden(input_embs_all.size(0))
        final_emb_list = []
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            final_emb_list.append(output.squeeze())
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        
        final_output = torch.stack(final_emb_list).transpose(0, 1)

        return torch.cat((input_embs_all, final_output), dim=2)


    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h0
    
class LLMGRU4rec(nn.Module):
    def __init__(self, config, final_act='tanh'):
        super(LLMGRU4rec, self).__init__()
        self.config = config
        self.hidden_size = self.config["hidden_units"]
        self.output_size = self.config["hidden_units"]
        self.num_layers = self.config["num_blocks"]
        self.dropout_hidden = self.config["dropout_rate"]
        self.dropout_input = self.config["dropout_rate"]
        self.embedding_dim = self.config["hidden_units"]
        self.batch_size = self.config["batch_size"]
        self.device = self.config["device"]
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                 item_embedding_dim=config["hidden_units"],
                                 layers=[config["hidden_units"]] * (config["dnn_layer"] + 1),
                                 drop_rate=config["dropout_rate"])

        self.create_final_activation(final_act)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward_step(self, input, hidden):
        output, hidden = self.gru(input, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def forward(self, interaction_list, interaction_mask, neg_list):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.fc(neg_list)[:, :-1, :]
        hidden = self.init_hidden(input_embs_all.size(0))
        final_emb_list = []
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            final_emb_list.append(output.squeeze())
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        
        final_output = torch.stack(final_emb_list).transpose(0, 1)

        return final_output[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, final_output), dim=2)


    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        item_embs = self.fc(item_indices)
        hidden = self.init_hidden(input_embs_all.size(0))
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        logits = item_embs.matmul(output.unsqueeze(-1)).squeeze(-1)

        return logits
    
    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)
        hidden = self.init_hidden(input_embs_all.size(0))
        final_emb_list = []
        for idx in range(input_embs_all.size(1)):
            output, hidden = self.forward_step(input_embs_all[:, idx, :].unsqueeze(0), hidden)
            final_emb_list.append(output.squeeze())
            mask = (log_mask[:, idx].squeeze() != 0.0).unsqueeze(0).unsqueeze(-1).expand_as(hidden)
            hidden *= mask
        final_output = torch.stack(final_emb_list).transpose(0, 1)

        return torch.cat((input_embs_all, final_output), dim=2)

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return h0
    

class MLP_Layers(torch.nn.Module):
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
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items):
        sample_items = self.activate(self.fc(sample_items))
        return self.mlp_layers(sample_items)