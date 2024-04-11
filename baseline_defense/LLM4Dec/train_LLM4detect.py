import json
import os
import random
import time
from utls.mydataset import SASDataset
from baseline_defense.LLM4Dec.Detector import Detector

from meta_config import args
from utls.model_config import *
from utls.utilize import init_run, restore_stdout_stderr


from datetime import datetime
import torch
import torch.optim as optim

from utls.utilize import calculate_f1, custom_loss, slice_lists


class Trainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self._create_dataset(f"data/{trainer_config['dataset']}")
        self._create_model()

    
    def _create_dataset(self, path):
        self.dataset = SASDataset(path, self.config["LLM"], self.config, has_fake_user=self.config["with_fd"], max_len=self.config["max_interaction"])

    def _create_model(self):
        self.fd_model = Detector(self.config["fd_config"]).to(self.device)
    
        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.fd_model.cuda()
    
        self.opt_fd = optim.AdamW(self.fd_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    def train(self, path=None):
        patience = self.config["patience"]
        best_metrics = -1
        best_model_path = path
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{datetime.now().strftime('%Y%m%d%H%M')}.pth")

        for epoch in range(self.n_epochs):
            start_t = time.time()
            epoch_loss_fd = 0
            all_fake = list(range(0, self.dataset.n_fake_users))
            all_user = list(range(0, self.dataset.n_users))
            user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
            for users, fakes in zip(user_list, fake_list):
                self.fd_model.train()
                self.opt_fd.zero_grad()
                _, _, label, detections = self.dataset.get_fake_user_batch(neg_idx=users, fk_idx=fakes)
                fd_predict = self.fd_model(detections)
                fd_loss = custom_loss(fd_predict, label, alpha=self.config["reg_entropy"], pos_weight_val=self.config["pos_weight"]) 
                fd_loss.backward()
                self.opt_fd.step()
                epoch_loss_fd += fd_loss.item()
            
            end_t = time.time()
            print(f"Epoch {epoch}: FD Loss: {epoch_loss_fd:.4f}, Time: {end_t-start_t:.2f}")
        
            if (epoch + 1) % self.config["val_interval"] == 0:
                avg_metrics = self._eval_fd(epoch)
                if avg_metrics > best_metrics:
                    best_metrics = avg_metrics
                    # Save the best model
                    # self._save_model(best_model_path)
                    torch.save({
                        'fd_model': self.fd_model.state_dict(),
                    }, best_model_path)
                    patience = self.config["patience"]
                else:
                    patience -= self.config["val_interval"]
                    if patience <= 0:
                        print('Early stopping!')
                        break
            
        checkpoint = torch.load(best_model_path)
        self.fd_model.load_state_dict(checkpoint['fd_model'])

        self._record_weight()
            
    
    def _record_weight(self):
        user_weight_dict = {}
        all_fake = list(range(0, self.dataset.n_fake_users))
        all_user = list(range(0, self.dataset.n_users))
        user_list, fake_list = slice_lists(all_user, all_fake, self.config["batch_size"])
        for users, fakes in zip(user_list, fake_list):
            self.fd_model.eval()
            _, _, label, detections = self.dataset.get_fake_user_batch(neg_idx=users, fk_idx=fakes)
            with torch.no_grad():
                fd_predict = self.fd_model(detections)
            scores = fd_predict[label == 0]
            for u, w in zip(users, scores):
                if w >= 0.5:
                    user_weight_dict[u] = 0.0
                else:
                    user_weight_dict[u] = 1.0
        
        path = f"baseline_defense/LLM4Dec/{self.config['dataset']}/{self.config['LLM']}/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.config["inject_user"]:
            with open(os.path.join(path, f'{self.config["inject_persent"]}_weight.json'), 'w') as f:
                json.dump(user_weight_dict, f)
        else:
            with open(os.path.join(path, f'normal_weight.json'), 'w') as f:
                json.dump(user_weight_dict, f)
        


    def _eval_fd(self, epoch):
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
            _, _, label, detection = self.dataset.get_fake_user_batch(neg_idx=users, final_test=1, fk_idx=fakes, inj_idx=injs)
            self.fd_model.eval()
            with torch.no_grad():
                fd_predict = self.fd_model(detection)
            tp, fp, fn = calculate_f1(fd_predict, label.to(self.device))
            tp_all += tp
            fp_all += fp
            fn_all += fn
        precision = tp_all / (tp_all + fp_all + 1e-6)
        recall = tp_all / (tp_all + fn_all + 1e-6)

        print(f"Epoch {epoch}: Precision: {precision:.4f}, Recall: {recall:.2f}")

        return 2 * precision * recall / (precision + recall + 1e-6)
    

def main(seed=2023):
    args.seed = seed
    path = f"./baseline_defense/LLM4Dec/log/{args.dataset}/{args.LLM}/"
    init_run(log_path=path, args=args, seed=args.seed)
    glo = globals()
    global_config = vars(args)

    print(global_config)
    global_config["fd_config"] = glo["get_FD_adapter_config"](global_config)
    print("FD Adapter")
    trainer = Trainer(global_config)
    trainer.train(f"./baseline_defense/LLM4Dec/checkpoints/{args.dataset}/{args.LLM}/")
    restore_stdout_stderr()

if __name__ == '__main__':
    main(seed=2023)
