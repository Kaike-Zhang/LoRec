import json
import random
import time
from baseline_defense.CL4rec.model import BasicCL4rec, LLMCL4rec
import torch
import torch.optim as optim
from backbone_model.SASrec.model import BasicSASrec, LLMSASrec
from backbone_model.FMLPrec.model import BasicFMLPrec, LLMFMLPrec
from backbone_model.GRU4rec.model import BasicGRU4rec, LLMGRU4rec
from baseline_defense.GraphRfi.NRF_detection import NeuralRandomForest
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle as pkl

from utls.trainer import SASrecTrainer
from utls.utilize import calculate_f1, custom_loss, slice_lists


class GraphRfi4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.opt_fd = optim.AdamW(self.fd_model.parameters(), lr=0.01, weight_decay=5e-4)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)

        self.fd_model = NeuralRandomForest(self.config["baseline_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
            self.fd_model.cuda()

    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        epoch_loss_fd = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            self.fd_model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, _ = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            with torch.no_grad():
                weight = self.fd_model(torch.mean(predict_emb, dim=1), torch.mean((1-torch.sigmoid(pos_logits)), dim=1))
            loss = self._get_rec_loss(pos_logits, neg_logits, mask, 1 - weight)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        epoch_loss_fd = self._train_fd_epoch(epoch)
        
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        loss_text += f", FD Loss: {epoch_loss_fd:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")
    

    def _train_fd_epoch(self, epoch):
        epoch_loss_fd = 0
        self.model.eval()
        self.fd_model.train()
        all_fake = list(range(0, self.dataset.n_fake_users))
        all_user = list(range(0, self.dataset.n_users))
        user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
        for users, fakes in zip(user_list, fake_list):
            self.opt_fd.zero_grad()
            emb, mask, label, _ = self.dataset.get_fake_user_batch(neg_idx=users, fk_idx=fakes)
            with torch.no_grad():
                predict_emb, target_pos_embs, _, _ = self.model(emb, mask, emb)
            logits = (predict_emb * target_pos_embs).sum(-1)
            fd_predict = self.fd_model(torch.mean(predict_emb, dim=1), torch.mean((1-logits), dim=1))
            
            fd_loss = self._get_fd_loss(fd_predict, label.unsqueeze(1).to(self.device).float())
            fd_loss.backward()
            self.opt_fd.step()
            epoch_loss_fd += fd_loss.item()
        
        return epoch_loss_fd

    
    
    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")

        if self.config["with_fd"]:
            precision, recall = self._eval_fd()
            print(f"Detection test - Precision: {precision:4f}, Recall: {recall:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr

    def _eval_fd(self):
        all_fake = list(range(0, self.dataset.n_fake_users))
        all_inj  = list(range(self.dataset.n_users - self.dataset.n_inject_user, self.dataset.n_users)) if self.config["inject_user"] else []

        if self.config["inject_user"]:
            all_user = list(range(0, self.dataset.n_users - self.dataset.n_inject_user))
            user_list, inj_list = slice_lists(all_user, all_inj, self.config["batch_size"])
            fake_list = [[] for _ in inj_list]
        else:
            all_user = list(range(0, self.dataset.n_users))
            user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
            inj_list = [[] for _ in fake_list]
            
        tp_all = 0
        fp_all = 0
        fn_all = 0
        for users, fakes, injs in zip(user_list, fake_list, inj_list):
            emb, mask, label, _ = self.dataset.get_fake_user_batch(neg_idx=users, final_test=1, fk_idx=fakes, inj_idx=injs)
            self.model.eval()
            self.fd_model.eval()
            with torch.no_grad():
                model_predict_emb, target_pos_embs, _, _ = self.model(emb, mask, emb)
                logits = (model_predict_emb * target_pos_embs).sum(-1)
            fd_predict = self.fd_model(torch.mean(model_predict_emb, dim=1), torch.mean((1-logits), dim=1))
            tp, fp, fn = calculate_f1(fd_predict, label.to(self.device))
            tp_all += tp
            fp_all += fp
            fn_all += fn
        precision = tp_all / (tp_all + fp_all + 1e-6)
        recall = tp_all / (tp_all + fn_all + 1e-6)
        return precision, recall
    
    def eval_weight(self, path):
        self._load_model(path)

        all_fake = list(range(0, self.dataset.n_fake_users))
        all_inj  = list(range(self.dataset.n_users - self.dataset.n_inject_user, self.dataset.n_users)) if self.config["inject_user"] else []

        if self.config["inject_user"]:
            all_user = list(range(0, self.dataset.n_users - self.dataset.n_inject_user))
            user_list, inj_list = slice_lists(all_user, all_inj, self.config["batch_size"])
            fake_list = [[] for _ in inj_list]
        else:
            all_user = list(range(0, self.dataset.n_users))
            user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
            inj_list = [[] for _ in fake_list]

        user_weight = 0
        inj_weight = 0
        for users, fakes, injs in zip(user_list, fake_list, inj_list):
            emb, mask, label, _ = self.dataset.get_fake_user_batch(neg_idx=users, final_test=1, fk_idx=fakes, inj_idx=injs)
            self.model.eval()
            self.fd_model.eval()
            with torch.no_grad():
                model_predict_emb, target_pos_embs, _, _ = self.model(emb, mask, emb)
                logits = (model_predict_emb * target_pos_embs).sum(-1)
            fd_predict = self.fd_model(torch.mean(model_predict_emb, dim=1), torch.mean((1-logits), dim=1))
            user_weight += (1 - fd_predict[label == 0]).sum()
            inj_weight += (1 - fd_predict[label == 1]).sum()
        return user_weight / (self.dataset.n_users - self.dataset.n_inject_user), inj_weight / self.dataset.n_inject_user

    def _get_fd_loss(self, fd_predict, label):
        return custom_loss(fd_predict, label.squeeze(), alpha=0, pos_weight_val=2.0)


class APR4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        self.adv_reg = trainer_config["baseline_config"]['adv_reg']
        self.eps = trainer_config["baseline_config"]['eps']
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)
        
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, _ = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            loss = self._get_rec_loss(pos_logits, neg_logits, mask)

            delta_users_r, delta_pos_items_r, delta_neg_items_r = \
                torch.autograd.grad(loss, (predict_emb, target_pos_embs, target_neg_embs), retain_graph=True)

            delta_users_r = F.normalize(delta_users_r, p=2, dim=1) * self.eps
            delta_pos_items_r = F.normalize(delta_pos_items_r, p=2, dim=1) * self.eps
            delta_neg_items_r = F.normalize(delta_neg_items_r, p=2, dim=1) * self.eps
            pos_logits = ((predict_emb + delta_users_r) * (target_pos_embs + delta_pos_items_r)).sum(-1)
            neg_logits = ((predict_emb + delta_users_r) * (target_neg_embs + delta_neg_items_r)).sum(-1)
            adv_loss = self._get_rec_loss(pos_logits, neg_logits, mask)

            loss += adv_loss
            
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
        
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")


    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr
    
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr, avg_ndcg
    
    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict()
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])
    

class ADV4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        self.adv_reg = trainer_config["baseline_config"]['adv_reg']
        self.eps = trainer_config["baseline_config"]['eps']
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)
        
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            loss = self._get_rec_loss(pos_logits, neg_logits, mask)
            loss.backward()
            self.opt.step()

            epoch_loss += loss.item()

            # only the text-based sequential recommender systems version
            seq_before = seq.clone()
            self.model.eval()
            with torch.enable_grad():
                for _ in range(1):
                    seq.requires_grad = True
                    predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
                    pos_logits = (predict_emb * target_pos_embs).sum(-1)
                    neg_logits = (predict_emb * target_neg_embs).sum(-1)
                    adv_loss = self._get_rec_loss(pos_logits, neg_logits, mask)
                    adv_loss.backward()
                    input_grad = seq.grad.data / (torch.norm(seq.grad.data, dim=-1, keepdim=True) + 1e-9)
                    seq = seq + self.eps * input_grad
                    seq = torch.clamp(seq, min=0.)
                    seq = seq / (seq.sum(-1, keepdim=True) + 1e-6)
                    seq = seq.detach()
            switch_indices = (torch.rand(seq_before.shape) <= 0.5).to(self.device)
            switch_indices = (switch_indices * (seq_before != 0)).float()
            seq = switch_indices * seq + (1 - switch_indices) * seq_before

            self.model.train()
            self.opt.zero_grad()
            predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            loss = self._get_rec_loss(pos_logits, neg_logits, mask)
            loss.backward()
            self.clip_gradients(5)
            self.opt.step()

            epoch_loss += loss.item()
        
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")
    
    def clip_gradients(self, limit=5):
        for p in self.model.parameters():
            torch.nn.utils.clip_grad_norm_(p, 5)

    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr
    
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr, avg_ndcg

    
    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict()
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])



class Denoise4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, _ = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            weight = F.cosine_similarity(predict_emb.reshape(-1, predict_emb.shape[-1]), target_pos_embs.reshape(-1, target_pos_embs.shape[-1]), dim=-1).reshape(pos_logits.shape)
            loss = self. _get_rec_loss(pos_logits, neg_logits, mask, weight)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")
    

    def _get_rec_loss(self, pos_logits, neg_logits, mask, weight):
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        indices = torch.where(torch.FloatTensor(mask)[:,:-1] != 0)
        expanded_fd_weight = torch.sigmoid(weight)
        
        # Calculate weighted BCE loss
        weighted_pos_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        weighted_neg_loss = self.bce_criterion(neg_logits[indices], neg_labels[indices])
        weighted_pos_loss = (weighted_pos_loss * expanded_fd_weight[indices]).mean()
        weighted_neg_loss = (weighted_neg_loss * expanded_fd_weight[indices]).mean()

        return weighted_pos_loss + weighted_neg_loss


    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr
    
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr, avg_ndcg
    
    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict()
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])



class CL4rec4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMCL4rec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicCL4rec(self.config["model_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        epoch_loss_fd = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            loss = self._get_rec_loss(pos_logits, neg_logits, mask)
            loss += self.config["model_config"]["lambda"] * self.model.crop_forward_loss(seq, mask)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        if self.config["with_fd"]:
            loss_text += f", FD Loss: {epoch_loss_fd/self.config['fd_minibatch']:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")


class Detection4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
        self.user_weight = self._load_weight()

    def _load_weight(self):
        with open(f'baseline_defense/Detection/{self.config["dataset"]}.pkl', 'rb') as f:
            weight = pkl.load(f)
        return {self.dataset.user_id[k]: v for k, v in weight.items()}
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            weight = torch.tensor([self.user_weight[idx] for idx in batch_data]).float()
            loss = self. _get_rec_loss(pos_logits, neg_logits, mask, weight)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")

    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr
    
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr, avg_ndcg
    
    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict()
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])




class LLM4Dec4SASrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)
    
        self.user_weight = self._load_weight()

    def _load_weight(self):
        if self.config["inject_user"]:
            with open(f'baseline_defense/LLM4Dec/{self.config["dataset"]}/{self.config["LLM"]}/{self.config["inject_persent"]}_weight.json', 'r') as f:
                weight = json.load(f)
        else:
            with open(f'baseline_defense/LLM4Dec/{self.config["dataset"]}/{self.config["LLM"]}/normal_weight.json', 'r') as f:
                weight = json.load(f)
        return {int(k): v for k, v in weight.items()}
    
    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, diff_emb = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            weight = torch.tensor([self.user_weight[idx.item()] for idx in batch_data]).float().to(self.device)
            loss = self. _get_rec_loss(pos_logits, neg_logits, mask, weight.unsqueeze(1))
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")


    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr
    
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")
        
        if self.config["inject_user"]:
            self.test_attack()

        return avg_hr, avg_ndcg

    
    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict()
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])



class LLM4Dec4FMLPrecTrainer(LLM4Dec4SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMFMLPrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicFMLPrec(self.config["model_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()


class LLM4Dec4GRU4recTrainer(LLM4Dec4SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMGRU4rec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicGRU4rec(self.config["model_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()