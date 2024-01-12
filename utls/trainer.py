import json
import os
import time
import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F
from models.LCT import LCTer
from utls.mydataset import FMLPDataset, SASDataset, GRU4recDataset
from backbone_model.SASrec.model import BasicSASrec, LLMSASrec
from backbone_model.FMLPrec.model import BasicFMLPrec, LLMFMLPrec
from backbone_model.GRU4rec.model import BasicGRU4rec, LLMGRU4rec
from utls.utilize import custom_loss, slice_lists


class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
    
    def _create_dataset(self, path):
        raise NotImplementedError
    
    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=True)

    def _create_model(self):
        raise NotImplementedError
    
    def _create_opt(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _eval_model(self, epoch):
        raise NotImplementedError
    
    def test_model(self):
        raise NotImplementedError

    def _save_model(self, best_model_path):
        raise NotImplementedError
    
    def _load_model(self, best_model_path):
        raise NotImplementedError

    def _update_weight(self):
        raise NotImplementedError


    def _init_path(self, path=None):
        if self.config["use_LLM"]: 
            best_model_path = f"{self.config['checkpoints']}/{self.config['model']}_{self.config['LLM']}/{self.config['dataset']}"
            if self.config["with_lct"]:
                best_model_path = f"{self.config['checkpoints']}/{self.config['model']}_{self.config['LLM']}/{self.config['dataset']}/FD"
        else:
            best_model_path = f"{self.config['checkpoints']}/{self.config['model']}_NoLLM/{self.config['dataset']}"
        if self.config["main_file"] != "":
            best_model_path = os.path.join(best_model_path, self.config["main_file"])
        if path is not None:
            best_model_path = path
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{'attack_'+ str(self.config['inject_persent']) if self.config['inject_user'] else 'normal'}_{datetime.now().strftime('%Y%m%d%H%M')}.pth")

        return best_model_path

    def _update_lct(self, epoch):
        if self.update_flag and self.config["with_lct"]:
            self.update_flag = False
            self._update_weight()
        if self.config['checkpoints'] == 'checkpoints':
            if (epoch + 1) > 30:
                if (epoch + 1) % (self.config["weight_update"] * 2) == 0 and self.config["with_lct"]:
                    self._update_weight()
            else:
                if (epoch + 1) % self.config["weight_update"] == 0 and self.config["with_lct"]:
                    self._update_weight()


    def train(self, path=None):
        if self.config["with_lct"]:
            lct_init = False
        patience = self.config["patience"]
        best_metrics = -1

        best_model_path = self._init_path(path=path)
        
        self.update_flag = False
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            self._update_lct(epoch)

            
            if (epoch + 1) % self.config["val_interval"] == 0:
                avg_metrics = self._eval_model(epoch)

                # Initialize patience after the start of LCT
                if self.config["with_lct"] and (epoch + 1) > self.config["lct_start"] and not lct_init:
                    patience = self.config["patience"]
                    best_metrics = -1
                    lct_init = True
                
                if (epoch + 1) >= self.config["min_epochs"]:
                    if avg_metrics > best_metrics:
                        best_metrics = avg_metrics
                        # Save the best model
                        self._save_model(best_model_path)
                        patience = self.config["patience"]
                    else:
                            patience -= self.config["val_interval"]
                            if patience <= 0:
                                print('Early stopping!')
                                break
                
        self._load_model(best_model_path)
        # Test
        hr, ndcg = self.test_model()

        return hr, ndcg


class SASrecTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_dataloader()
        self._create_model()
        self._create_opt()

    def _create_dataset(self, path):
        self.dataset = SASDataset(path, self.config["LLM"], self.config, has_fake_user=self.config["with_lct"], max_len=self.config["max_interaction"])
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMSASrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicSASrec(self.config["model_config"]).to(self.device)

        self.lct_model = LCTer(self.config["lct_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
            self.lct_model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.opt_lct = optim.AdamW(self.lct_model.parameters(), lr=0.01, weight_decay=5e-4)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        

    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        epoch_loss_lct = 0

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            seq, mask, neg = self.dataset.get_train_batch(batch_data)
            predict_emb, target_pos_embs, target_neg_embs, _ = self.model(seq, mask, neg)
            pos_logits = (predict_emb * target_pos_embs).sum(-1)
            neg_logits = (predict_emb * target_neg_embs).sum(-1)
            if self.config["with_lct"] and (epoch + 1) >= self.config["lct_start"]:
                weight = self.dataset.get_weight(batch_data)
                loss = self. _get_rec_loss(pos_logits, neg_logits, mask, weight)
            else:
                loss = self._get_rec_loss(pos_logits, neg_logits, mask)
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        if self.config["with_lct"]:
            for _ in range(self.config["lct_minibatch"]):
                epoch_loss_lct += self._train_lct_epoch(epoch)
        
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        if self.config["with_lct"]:
            loss_text += f", LCT Loss: {epoch_loss_lct/self.config['lct_minibatch']:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")


    def _train_lct_epoch(self, epoch):
        epoch_loss_lct = 0
        self.model.eval()
        self.lct_model.train()
        all_fake = list(range(0, self.dataset.n_fake_users))
        all_user = list(range(0, self.dataset.n_users))
        user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
        for users, fakes in zip(user_list, fake_list):
            self.opt_lct.zero_grad()
            emb, mask, label, detections = self.dataset.get_fake_user_batch(neg_idx=users, fk_idx=fakes)
            with torch.no_grad():
                model_predict_emb = self.model.get_emb(emb, mask)
            lct_predict, similarity = self.lct_model(seq=model_predict_emb, detections=detections, mask=mask)
            lct_loss = self._get_lct_loss(lct_predict, label, similarity)
            lct_loss.backward()
            self.opt_lct.step()
            epoch_loss_lct += lct_loss.item()

        return epoch_loss_lct

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
    

    def _eval_rec(self, org_seq, seq, mask, pos):
        self.model.eval()
        all_idx = list(range(self.dataset.n_items+1))
        if self.config["use_LLM"]:
            all_idx = self.dataset.get_labels_emb(all_idx)
        with torch.no_grad():
            all_logits = self.model.predict(seq, mask, all_idx)
        for i in range(all_logits.size(0)):
            all_logits[i, org_seq[i]] = float('-inf')
        all_logits = all_logits[:,1:]
        HR = 0
        NDCG = 0
        _, sorted_indices = all_logits.sort(dim=1, descending=True)
        for user_idx, item_idx in enumerate(pos):
            rank = (sorted_indices[user_idx] == item_idx-1).nonzero().item() + 1
            if rank <= self.config["top_k"]:
                HR += 1
                NDCG += 1 / np.log2(rank + 1)
        
        return HR, NDCG
    
    def _update_weight(self):
        all_user = list(range(0, self.dataset.n_users))
        all_socre = []
        user_score_dict = {}
        user_list = [all_user[i:i + self.config["batch_size"]] for i in range(0, len(all_user), self.config["batch_size"])]
        fake_list = [[] for _ in user_list]
        for users, fakes in zip(user_list, fake_list):
            self.model.eval()
            self.lct_model.eval()
            emb, mask, label, _ = self.dataset.get_fake_user_batch(neg_idx=users, final_test=0, fk_idx=fakes)
            with torch.no_grad():
                model_predict_emb = self.model.get_emb(emb, mask)
                score, _ = self.lct_model(seq=model_predict_emb, mask=mask)
            score_list = score.tolist()
            all_socre.extend(score_list)
            for user, user_score in zip(users, score_list):
                user_score_dict[user] = user_score
        mean_score = np.mean(all_socre)
        up_idx = []
        for user, score in user_score_dict.items():
            if score > mean_score:
                up_idx.append(user)
        if len(up_idx) < len(user_score_dict) / 3:
            self.dataset.update_weight(up_idx)
        else:
            self.update_flag = True


    def _save_model(self, best_model_path):
        torch.save({
            'SASrec': self.model.state_dict(),
            'lct_model': self.lct_model.state_dict(),
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['SASrec'])
        self.lct_model.load_state_dict(checkpoint['lct_model'])

    def _get_rec_loss(self, pos_logits, neg_logits, mask, lct_weight=None):
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        indices = torch.where(torch.FloatTensor(mask)[:,:-1] != 0)
        if lct_weight is None:
            weighted_pos_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices]).mean()
            weighted_neg_loss = self.bce_criterion(neg_logits[indices], neg_labels[indices]).mean()
        else:
            expanded_lct_weight = lct_weight.expand_as(pos_logits)[indices]
            
            # Calculate weighted BCE loss
            weighted_pos_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
            weighted_neg_loss = self.bce_criterion(neg_logits[indices], neg_labels[indices])
            weighted_pos_loss = (weighted_pos_loss * expanded_lct_weight).mean()
            weighted_neg_loss = (weighted_neg_loss * expanded_lct_weight).mean()

        return weighted_pos_loss + weighted_neg_loss
    
    def _get_lct_loss(self, lct_predict, label, similarity):
        return custom_loss(lct_predict, label, alpha=self.config["reg_entropy"]) - self.config["sim_weight"] * similarity

    def test_attack(self, path=None):
        def add_list(list_1, list_2):
            return list(map(lambda x, y: x + y, list_1, list_2))
        def div_list(list_1, list_2):
            return list(map(lambda x, y: x / y, list_1, list_2))
        if path is not None:
            self._load_model(path)
        targe_item = self.dataset.target_item
        avg_hr = [0] * len(targe_item)
        avg_ndcg = [0] * len(targe_item)
        avg_rank = [0] * len(targe_item)
        all_cnt = [0] * len(targe_item)
        self.model.eval()
        for batch_data in self.dataloader:
            org_seq, seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            hr, ndcg, rank, cnt = self._eval_taget(org_seq, seq, mask, pos, targe_item)
            avg_hr = add_list(avg_hr, hr)
            avg_ndcg = add_list(avg_ndcg, ndcg)
            avg_rank = add_list(avg_rank, rank)
            all_cnt = add_list(all_cnt, cnt)
        print(f"Normal Model: Target Item - HR@{self.config['attack_top_k']}:{np.mean(div_list(avg_hr, all_cnt)):4f};  NDCG@{self.config['attack_top_k']}:{np.mean(div_list(avg_ndcg, all_cnt)):4f}; RANK:{np.mean(div_list(avg_rank, all_cnt)):4f}")

        return np.mean(div_list(avg_hr, all_cnt)), np.mean(div_list(avg_ndcg, all_cnt))

    def _eval_taget(self, org_seq, seq, mask, pos, targe_item):
        hr_item = [0] * len(targe_item)
        ndcg_item = [0] * len(targe_item)
        cnt = [0] * len(targe_item)
        rank_item = [0] * len(targe_item)
        all_idx = list(range(self.dataset.n_items+1))
        if self.config["use_LLM"]:
            all_idx = self.dataset.get_labels_emb(all_idx)
        with torch.no_grad():
            all_logits = self.model.predict(seq, mask, all_idx)
        for i in range(all_logits.size(0)):
            all_logits[i, org_seq[i]] = float('-inf')
        all_logits = all_logits[:,1:]
        _, sorted_indices = all_logits.sort(dim=1, descending=True)
        for user in range(len(pos)):
            for id, item in enumerate(targe_item):
                if item in org_seq[i]:
                    continue
                rank = (sorted_indices[user] == item-1).nonzero().item() + 1
                rank_item[id] += rank
                cnt[id] += 1
                if rank <= self.config["attack_top_k"]:
                    hr_item[id] += 1
                    ndcg_item[id] += 1 / np.log2(rank + 1)
        
        return hr_item, ndcg_item, rank_item, cnt


class FMLPrecTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_dataset(self, path):
        self.dataset = FMLPDataset(path, self.config["LLM"], self.config, has_fake_user=self.config["with_lct"], max_len=self.config["max_interaction"])
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMFMLPrec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicFMLPrec(self.config["model_config"]).to(self.device)

        self.lct_model = LCTer(self.config["lct_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
            self.lct_model.cuda()


class GRU4recTrainer(SASrecTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

    def _create_dataset(self, path):
        self.dataset = GRU4recDataset(path, self.config["LLM"], self.config, has_fake_user=self.config["with_lct"], max_len=self.config["max_interaction"])
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        if self.config["use_LLM"]:
            self.model = LLMGRU4rec(self.config["model_config"]).to(self.device)
        else:
            self.model = BasicGRU4rec(self.config["model_config"]).to(self.device)

        self.lct_model = LCTer(self.config["lct_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
            self.lct_model.cuda()