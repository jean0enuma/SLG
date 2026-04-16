from mpmath.functions.zeta import polylog_series
from networkx.utils import powerlaw_sequence
from transformers.training_args import trainer_log_levels

from trainer.based_trainer import BaseTrainer
import torch
from tqdm import tqdm
import numpy as np
import gc
import random
import pandas as pd
import wandb
import os
import cv2
import torch.nn.functional as F
import shutil
from models.module.EMA import EMA
import matplotlib.pyplot as plt
from loader.coordinate_preprocess import apply_savgol_filter,average_movint




def save_code_usage_histogram(
    hist_dict,
    save_path,
    top_k=None,
    title="Code Usage Histogram"
):
    """
    Save histogram (counts) to specified path.

    hist_dict: output of model.code_usage_histogram()
    save_path: full file path (e.g., "/mnt/data/hist.png")
    top_k: if not None, save only top_k most frequent codes
    """
    counts = hist_dict["counts"].detach().cpu()

    if top_k is not None:
        values, indices = torch.topk(counts, k=top_k)
        x = indices.numpy()
        y = values.numpy()
    else:
        x = torch.arange(len(counts)).numpy()
        y = counts.numpy()

    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Code Index")
    plt.ylabel("Usage Count")
    plt.title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_code_usage_probability(
    hist_dict,
    save_path,
    top_k=None,
    title="Code Usage Probability"
):
    """
    Save probability histogram to specified path.
    """
    probs = hist_dict["probs"].detach().cpu()

    if top_k is not None:
        values, indices = torch.topk(probs, k=top_k)
        x = indices.numpy()
        y = values.numpy()
    else:
        x = torch.arange(len(probs)).numpy()
        y = probs.numpy()

    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Code Index")
    plt.ylabel("Probability")
    plt.title(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)




class CLIP_trainer:
    def __init__(self,config,scheduler=None):
        self.config=config
        self.scheduler=scheduler
        self.step=0
    def generate_scheduler(self,epoch):
        #学習中にteacher forcing率を変化させるスケジューラを生成
        #エポック単位でcosineで0から1へ変化させる
        total_epochs=self.config["lr_parameters"]['epoch']
        def scheduler(epoch):
            return 1-0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        if epoch >total_epochs:
            self.g_scheduler=1.0
        else:
            self.g_scheduler=scheduler(epoch)
    def train(self, model, optimizer,train_loader, device,ema=False):
        model.train()
        total_loss = []
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["lr_parameters"]["amp"])
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader.dataset) // train_loader.batch_size):
            padded_cod_data,padded_mask, input_length_tensor, id_list,data_path,sequence=batch
            padded_cod_data=padded_cod_data.float().to(device)
            padded_mask=padded_mask.to(device)
            input_length_tensor=input_length_tensor.to(device)
            id_list=id_list.to(device)
            sequence=sequence.to(device)
            input_ids=sequence['input_ids'].to(device)
            attention_mask=sequence['attention_mask'].to(device)
            token_type_ids=sequence['token_type_ids'].to(device) if 'token_type_ids' in sequence.keys() else None
            optimizer.zero_grad(set_to_none=True)
            #g_prob = random.random()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                loss_dict = model(padded_cod_data, input_ids,attention_mask,input_length_tensor,token_type_ids)

            loss = loss_dict['loss']
            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f"Warning: Loss is {loss.item()} at batch {batch_idx}. Skipping this batch.")
                continue
            scaler.scale(loss).backward()
            ##grad_clip
            if self.config["lr_parameters"]["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["lr_parameters"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update()
            if self.scheduler is not None and self.config["lr_parameters"]["scheduler_timing"] == "step":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(self.step)
                else:
                    self.scheduler.step()
                self.step += 1
            total_loss.append(loss.item())

            # codebook replacement
            if batch_idx % 10== 0:
                tqdm.write(f"Avg Loss: {np.mean(total_loss)}")

        avg_loss = np.mean(total_loss).astype(np.float32)
        return {
            "loss": avg_loss,
        }
    def eval(self, model,  test_loader, device):
        model.eval()
        total_loss = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader.dataset) // test_loader.batch_size):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path,sequence = batch
                padded_cod_data = padded_cod_data.float().to(device)
                padded_mask = padded_mask.to(device)
                input_length_tensor = input_length_tensor.to(device)
                sequence=sequence.to(device)
                input_ids=sequence['input_ids'].to(device)
                attention_mask=sequence['attention_mask'].to(device)
                token_type_ids=sequence['token_type_ids'].to(device) if 'token_type_ids' in sequence.keys() else None
                id_list = id_list.to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                    loss_dict = model(padded_cod_data, input_ids,attention_mask,input_length_tensor,token_type_ids)
                loss = loss_dict['loss']
                total_loss.append(loss.item())

        avg_loss = np.mean(total_loss).astype(np.float32)
        return {
            "loss": avg_loss,
        }
    def fit(self,model,optimizer,scheduler,train_loader,eval_loader,test_loader,device,early_stopping=None):
        if self.config["lr_parameters"]["ema"]:
            ema_model=EMA(model,self.config["lr_parameters"]["ema_beta"])
        num_epochs=self.config["lr_parameters"]['epoch']
        train_loss_list=self.config['train_loss_list'] if 'train_loss_list' in self.config.keys() else []
        eval_loss_list=self.config['eval_loss_list'] if 'eval_loss_list' in self.config.keys() else []
        test_loss_list=self.config['test_loss_list'] if 'test_loss_list' in self.config.keys() else []
        save_path=self.config["save_path"]
        for epoch in range(self.config["init_epoch"], num_epochs):
            #self.generate_scheduler(epoch)
            print(f"saved path:{save_path}")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"base_lr:{scheduler.get_last_lr()}")
            print(f"epoch:{epoch}/{self.config['lr_parameters']['epoch']}")
            os.makedirs(f"{save_path}/{epoch}", exist_ok=True)
            print("--train--")
            train_loss = self.train(model, optimizer,  train_loader, device,ema=self.config["lr_parameters"]["ema"])
            print("--eval--")
            eval_loss = self.eval(model, eval_loader, device)
            print("--test--")
            test_loss = self.eval(model, test_loader, device)
            train_loss_list.append(train_loss['loss'])

            eval_loss_list.append(eval_loss['loss'])

            test_loss_list.append(test_loss['loss'])

            print(f"Epoch {epoch+1}/{num_epochs})")
            print(f"Train Loss: {train_loss['loss']:.4f}")
            print(f"Eval Loss: {eval_loss['loss']:.4f}")
            print(f"Test Loss: {test_loss['loss']:.4f}")
            #eval_lossとtest_lossのkeyを変更
            eval_loss = {
                "eval_loss": eval_loss['loss'],
            }
            test_loss = {
                "test_loss": test_loss['loss'],
            }
            log_dict={**train_loss,**eval_loss,**test_loss}
            wandb.log(log_dict)
            torch.save(model.state_dict(), f"{save_path}/{epoch}/model_epoch{epoch}.pth")
            if self.config["lr_parameters"]["ema"]:
                torch.save(ema_model.ema_model.state_dict(), f"{save_path}/{epoch}/ema_model_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss_list": train_loss_list,
                "eval_loss_list": eval_loss_list,
                "test_loss_list": test_loss_list,
                'random': random.getstate(),
                'np_random': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_random': torch.get_rng_state(),
                'cuda_random': torch.cuda.get_rng_state(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }, f"{save_path}/checkpoint.cpt")
            log_data = pd.DataFrame(
                {
                    "epoch": list(range(epoch + 1)),
                    "train_loss": train_loss_list,
                    "eval_loss": eval_loss_list,
                    "test_loss": test_loss_list,
                }
            )
            log_data.to_csv(f"{save_path}/log.csv")
            if self.scheduler is not None and self.config["lr_parameters"]["scheduler_timing"] == "epoch":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(epoch + 1)
                else:
                    self.scheduler.step()
            # 早期終了のチェック
            if early_stopping:
                early_stopping(eval_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        wandb.alert(
            title="Finish",
            text='無事学習が終了しました。'
        )

        return






