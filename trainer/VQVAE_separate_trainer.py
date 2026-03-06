from mpmath.functions.zeta import polylog_series
from networkx.utils import powerlaw_sequence

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




class VQVAESeparateTrainer(BaseTrainer):
    def __init__(self,config,scheduler=None):
        self.config=config
        self.scheduler=scheduler
        self.step=0
        self.g_scheduler=0.0
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
    def compute_loss(self, batch, model, criterion):
        padded_cod_data, padded_mask,input_length_tensor, id_list = batch

        loss = model(padded_cod_data,hand_valid_mask=padded_mask,input_length=input_length_tensor)
        return loss
    def train(self, model, optimizer, criterion, train_loader, device,ema=False):
        model.train()
        total_loss = []
        total_recon_pose_loss=[]
        total_recon_hand_loss=[]
        total_recon_extra_loss=[]
        total_vq_loss=[]
        total_hand_vq_loss=[]
        total_extra_vq_loss=[]
        total_perplexity=[]
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["lr_parameters"]["amp"])
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader.dataset) // train_loader.batch_size):
            padded_cod_data,padded_mask, input_length_tensor, id_list,data_path=batch
            padded_cod_data=padded_cod_data.float().to(device)
            padded_mask=padded_mask.to(device)
            input_length_tensor=input_length_tensor.to(device)
            id_list=id_list.to(device)
            batch = (padded_cod_data,padded_mask, input_length_tensor, id_list)
            optimizer.zero_grad(set_to_none=True)
            #g_prob = random.random()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                loss_dict = self.compute_loss(batch, model, criterion)

            loss = loss_dict['loss_total']
            recon_pose_loss = loss_dict['pose_recon_loss']
            recon_hand_loss = loss_dict['hand_recon_loss']
            recon_extra_loss = loss_dict['extra_recon_loss']
            pose_vq_loss = loss_dict['pose_vq_loss']
            hand_vq_loss = loss_dict['hand_vq_loss']
            extra_vq_loss = loss_dict['extra_vq_loss']
            perplexity = loss_dict['perplexity']
            scaler.scale(loss).backward()
            ##grad_clip
            if self.config["lr_parameters"]["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["lr_parameters"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update()
            total_loss.append(loss.item())
            total_recon_pose_loss.append(recon_pose_loss.item())
            total_recon_hand_loss.append(recon_hand_loss.item())
            total_recon_extra_loss.append(recon_extra_loss.item())
            total_vq_loss.append(pose_vq_loss.item())
            total_hand_vq_loss.append(hand_vq_loss.item())
            total_extra_vq_loss.append(extra_vq_loss.item())
            total_perplexity.append(perplexity.item())
            if batch_idx % 100== 0:
                tqdm.write(f"Avg Loss: {np.mean(total_loss)}")
                tqdm.write(f"Avg Recon Pose Loss: {np.mean(total_recon_pose_loss)}")
                tqdm.write(f"Avg Recon Hand Loss: {np.mean(total_recon_hand_loss)}")
                tqdm.write(f"Avg Recon Extra Loss: {np.mean(total_recon_extra_loss)}")
                tqdm.write(f"Avg VQ Loss: {np.mean(total_vq_loss)}")
                tqdm.write(f"Avg Hand VQ Loss: {np.mean(total_hand_vq_loss)}")
                tqdm.write(f"Avg Extra VQ Loss: {np.mean(total_extra_vq_loss)}")
                tqdm.write(f"Avg Perplexity: {np.mean(total_perplexity)}")

        avg_loss = np.mean(total_loss).astype(np.float32)
        recon_pose_avg_loss = np.mean(total_recon_pose_loss).astype(np.float32)
        recon_hand_avg_loss = np.mean(total_recon_hand_loss).astype(np.float32)
        recon_extra_avg_loss = np.mean(total_recon_extra_loss).astype(np.float32)
        vq_avg_loss = np.mean(total_vq_loss).astype(np.float32)
        hand_vq_avg_loss = np.mean(total_hand_vq_loss).astype(np.float32)
        extra_vq_avg_loss = np.mean(total_extra_vq_loss).astype(np.float32)
        perplexity_avg = np.mean(total_perplexity).astype(np.float32)
        return {
            "loss": avg_loss,
            "recon_pose_loss": recon_pose_avg_loss,
            "recon_dir_loss": recon_hand_avg_loss,
            "vel_loss": recon_extra_avg_loss,
            "vq_loss": vq_avg_loss,
            "hand_vq_loss": hand_vq_avg_loss,
            "extra_vq_loss": extra_vq_avg_loss,
            "perplexity": perplexity_avg
        }
    def eval(self, model, criterion, test_loader, device):
        model.eval()
        total_loss = []
        total_recon_pose_loss=[]
        total_recon_hand_loss=[]
        total_recon_extra_loss=[]
        total_vq_loss=[]
        total_hand_vq_loss=[]
        total_extra_vq_loss=[]
        total_perplexity=[]
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader.dataset) // test_loader.batch_size):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                padded_cod_data = padded_cod_data.float().to(device)
                padded_mask = padded_mask.to(device)
                input_length_tensor = input_length_tensor.to(device)
                id_list = id_list.to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                    loss_dict = self.compute_loss(batch, model, criterion)
                loss = loss_dict['loss_total']
                recon_pose_loss = loss_dict['pose_recon_loss']
                recon_hand_loss = loss_dict['hand_recon_loss']
                recon_extra_loss = loss_dict['extra_recon_loss']
                pose_vq_loss = loss_dict['pose_vq_loss']
                hand_vq_loss = loss_dict['hand_vq_loss']
                extra_vq_loss = loss_dict['extra_vq_loss']
                perplexity = loss_dict['perplexity']
                total_loss.append(loss.item())
                total_recon_pose_loss.append(recon_pose_loss.item())
                total_recon_hand_loss.append(recon_hand_loss.item())
                total_recon_extra_loss.append(recon_extra_loss.item())
                total_vq_loss.append(pose_vq_loss.item())
                total_hand_vq_loss.append(hand_vq_loss.item())
                total_extra_vq_loss.append(extra_vq_loss.item())
                total_perplexity.append(perplexity.item())
        avg_loss = np.mean(total_loss).astype(np.float32)
        recon_pose_avg_loss=np.mean(total_recon_pose_loss).astype(np.float32)
        recon_hand_avg_loss=np.mean(total_recon_hand_loss).astype(np.float32)
        recon_extra_avg_loss=np.mean(total_recon_extra_loss).astype(np.float32)
        vq_avg_loss=np.mean(total_vq_loss).astype(np.float32)
        hand_vq_avg_loss=np.mean(total_hand_vq_loss).astype(np.float32)
        extra_vq_avg_loss=np.mean(total_extra_vq_loss).astype(np.float32)
        perplexity_avg=np.mean(total_perplexity).astype(np.float32)
        return {
            "loss": avg_loss,
            "recon_pose_loss": recon_pose_avg_loss,
            "recon_dir_loss": recon_hand_avg_loss,
            "vel_loss": recon_extra_avg_loss,
            "vq_loss": vq_avg_loss,
            "hand_vq_loss": hand_vq_avg_loss,
            "extra_vq_loss": extra_vq_avg_loss,
            "perplexity": perplexity_avg
        }
    def fit(self,model,optimizer,scheduler,criterion,train_loader,eval_loader,test_loader,device,early_stopping=None):
        if self.config["lr_parameters"]["ema"]:
            ema_model=EMA(model,self.config["lr_parameters"]["ema_beta"])
        num_epochs=self.config["lr_parameters"]['epoch']
        train_loss_list=self.config['train_loss_list'] if 'train_loss_list' in self.config.keys() else []
        train_recon_pose_loss_list=self.config['train_recon_pose_loss_list'] if 'train_recon_pose_loss_list' in self.config.keys() else []
        train_recon_dir_loss_list=self.config['train_recon_dir_loss_list'] if 'train_recon_dir_loss_list' in self.config.keys() else []
        train_recon_extra_loss_list=self.config['train_recon_extra_loss_list'] if 'train_recon_extra_loss_list' in self.config.keys() else []
        train_vq_loss_list=self.config['train_vq_loss_list'] if 'train_vq_loss_list' in self.config.keys() else []
        train_vq_hand_loss_list=self.config['train_vq_hand_loss_list'] if 'train_vq_hand_loss_list' in self.config.keys() else []
        train_vq_extra_loss_list=self.config['train_vq_extra_loss_list'] if 'train_vq_extra_loss_list' in self.config.keys() else []
        train_perplexity_list=self.config['train_perplexity_list'] if 'train_perplexity_list' in self.config.keys() else []
        eval_loss_list=self.config['eval_loss_list'] if 'eval_loss_list' in self.config.keys() else []
        eval_recon_pose_loss_list=self.config['eval_recon_pose_loss_list'] if 'eval_pose_loss_list' in self.config.keys() else []
        eval_recon_dir_loss_list=self.config['eval_recon_dir_loss_list'] if 'eval_recon_dir_loss_list' in self.config.keys() else []
        eval_recon_extra_loss_list=self.config['eval_recon_extra_loss_list'] if 'eval_recon_extra_loss_list' in self.config.keys() else []
        eval_recon_vq_loss_list=self.config['eval_recon_vq_loss_list'] if 'eval_recon_vq_loss_list' in self.config.keys() else []
        eval_recon_vq_hand_loss_list=self.config['eval_recon_vq_hand_loss_list'] if 'eval_recon_vq_hand_loss_list' in self.config.keys() else []
        eval_recon_vq_extra_loss_list=self.config['eval_recon_vq_extra_loss_list'] if 'eval_recon_vq_extra_loss_list' in self.config.keys() else []
        eval_recon_perplexity_list=self.config['eval_recon_perplexity_list'] if 'eval_recon_perplexity_list' in self.config.keys() else []
        test_loss_list=self.config['test_loss_list'] if 'test_loss_list' in self.config.keys() else []
        test_recon_pose_loss_list=self.config['test_recon_pose_loss_list'] if 'test_recon_pose_loss_list' in self.config.keys() else []
        test_recon_dir_loss_list=self.config['test_recon_dir_loss_list'] if 'test_recon_dir_loss_list' in self.config.keys() else []
        test_recon_extra_loss_list=self.config['test_recon_extra_loss_list'] if 'test_recon_extra_loss_list' in self.config.keys() else []
        test_recon_vq_loss_list=self.config['test_recon_vq_loss_list'] if 'test_recon_vq_loss_list' in self.config.keys() else []
        test_recon_vq_hand_loss_list=self.config['test_recon_vq_hand_loss_list'] if 'test_recon_vq_hand_loss_list' in self.config.keys() else []
        test_recon_vq_extra_loss_list=self.config['test_recon_vq_extra_loss_list'] if 'test_recon_vq_extra_loss_list' in self.config.keys() else []
        test_recon_perplexity_list=self.config['test_recon_perplexity_list'] if 'test_recon_perplexity_list' in self.config.keys() else []
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
            train_loss = self.train(model, optimizer, criterion, train_loader, device,ema=self.config["lr_parameters"]["ema"])
            print("--eval--")
            eval_loss = self.eval(model, criterion, eval_loader, device)
            print("--test--")
            test_loss = self.eval(model, criterion, test_loader, device)
            train_loss_list.append(train_loss['loss'])
            train_recon_pose_loss_list.append(train_loss['recon_pose_loss'])
            train_recon_dir_loss_list.append(train_loss['recon_dir_loss'])
            train_recon_extra_loss_list.append(train_loss['vel_loss'])
            train_vq_loss_list.append(train_loss['vq_loss'])
            train_vq_hand_loss_list.append(train_loss['hand_vq_loss'])
            train_vq_extra_loss_list.append(train_loss['extra_vq_loss'])
            train_perplexity_list.append(train_loss['perplexity'])
            eval_loss_list.append(eval_loss['loss'])
            eval_recon_pose_loss_list.append(eval_loss['recon_pose_loss'])
            eval_recon_dir_loss_list.append(eval_loss['recon_dir_loss'])
            eval_recon_extra_loss_list.append(eval_loss['vel_loss'])
            eval_recon_vq_loss_list.append(eval_loss['vq_loss'])
            eval_recon_vq_hand_loss_list.append(eval_loss['hand_vq_loss'])
            eval_recon_vq_extra_loss_list.append(eval_loss['extra_vq_loss'])
            eval_recon_perplexity_list.append(eval_loss['perplexity'])
            test_loss_list.append(test_loss['loss'])
            test_recon_pose_loss_list.append(test_loss['recon_pose_loss'])
            test_recon_dir_loss_list.append(test_loss['recon_dir_loss'])
            test_recon_extra_loss_list.append(test_loss['vel_loss'])
            test_recon_vq_loss_list.append(test_loss['vq_loss'])
            test_recon_vq_hand_loss_list.append(test_loss['hand_vq_loss'])
            test_recon_vq_extra_loss_list.append(test_loss['extra_vq_loss'])
            test_recon_perplexity_list.append(test_loss['perplexity'])
            print(f"Epoch {epoch+1}/{num_epochs})")
            print(f"Train Loss: {train_loss['loss']:.4f}, Recon Pose Loss: {train_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {train_loss['recon_dir_loss']:.4f}, Recon Extra Loss: {train_loss['vel_loss']:.4f}, VQ Loss: {train_loss['vq_loss']:.4f}, Hand VQ Loss: {train_loss['hand_vq_loss']:.4f}, Extra VQ Loss: {train_loss['extra_vq_loss']:.4f}, Perplexity: {train_loss['perplexity']:.4f}")
            print(f"Eval Loss: {eval_loss['loss']:.4f}, Recon Pose Loss: {eval_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {eval_loss['recon_dir_loss']:.4f}, Recon Extra Loss: {eval_loss['vel_loss']:.4f}, VQ Loss: {eval_loss['vq_loss']:.4f}, Hand VQ Loss: {eval_loss['hand_vq_loss']:.4f}, Extra VQ Loss: {eval_loss['extra_vq_loss']:.4f}, Perplexity: {eval_loss['perplexity']:.4f}")
            print(f"Test Loss: {test_loss['loss']:.4f}, Recon Pose Loss: {test_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {test_loss['recon_dir_loss']:.4f}, Recon Extra Loss: {test_loss['vel_loss']:.4f}, VQ Loss: {test_loss['vq_loss']:.4f}, Hand VQ Loss: {test_loss['hand_vq_loss']:.4f}, Extra VQ Loss: {test_loss['extra_vq_loss']:.4f}, Perplexity: {test_loss['perplexity']:.4f}")
            #eval_lossとtest_lossのkeyを変更
            eval_loss = {
                "eval_loss": eval_loss['loss'],
                "eval_recon_pose_loss": eval_loss['recon_pose_loss'],
                "eval_recon_dir_loss": eval_loss['recon_dir_loss'],
                "eval_recon_extra_loss": eval_loss['vel_loss'],
                "eval_vq_loss": eval_loss['vq_loss'],
                "eval_hand_vq_loss": eval_loss['hand_vq_loss'],
                "eval_extra_vq_loss": eval_loss['extra_vq_loss'],
                "eval_perplexity": eval_loss['perplexity']
            }
            test_loss = {
                "test_loss": test_loss['loss'],
                "test_recon_pose_loss": test_loss['recon_pose_loss'],
                "test_recon_dir_loss": test_loss['recon_dir_loss'],
                "test_recon_extra_loss": test_loss['vel_loss'],
                "test_vq_loss": test_loss['vq_loss'],
                "test_hand_vq_loss": test_loss['hand_vq_loss'],
                "test_extra_vq_loss": test_loss['extra_vq_loss'],
                "test_perplexity": test_loss['perplexity']
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
                "train_recon_pose_loss_list": train_recon_pose_loss_list,
                "train_recon_dir_loss_list": train_recon_dir_loss_list,
                "train_recon_extra_loss_list": train_recon_extra_loss_list,
                "train_vq_loss_list": train_vq_loss_list,
                "train_vq_hand_loss_list": train_vq_hand_loss_list,
                "train_vq_extra_loss_list": train_vq_extra_loss_list,
                "train_perplexity_list": train_perplexity_list,
                "eval_loss_list": eval_loss_list,
                "eval_recon_pose_loss_list": eval_recon_pose_loss_list,
                "eval_recon_dir_loss_list": eval_recon_dir_loss_list,
                "eval_recon_extra_loss_list": eval_recon_extra_loss_list,
                "eval_recon_vq_loss_list": eval_recon_vq_loss_list,
                "eval_recon_vq_hand_loss_list": eval_recon_vq_hand_loss_list,
                "eval_recon_vq_extra_loss_list": eval_recon_vq_extra_loss_list,
                "eval_recon_perplexity_list": eval_recon_perplexity_list,
                "test_loss_list": test_loss_list,
                "test_recon_pose_loss_list": test_recon_pose_loss_list,
                "test_recon_dir_loss_list": test_recon_dir_loss_list,
                "test_recon_extra_loss_list": test_recon_extra_loss_list,
                "test_recon_vq_loss_list": test_recon_vq_loss_list,
                "test_recon_vq_hand_loss_list": test_recon_vq_hand_loss_list,
                "test_recon_vq_extra_loss_list": test_recon_vq_extra_loss_list,
                "test_recon_perplexity_list": test_recon_perplexity_list,
                'random': random.getstate(),
                'np_random': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_random': torch.get_rng_state(),
                'cuda_random': torch.cuda.get_rng_state(),
            }, f"{save_path}/checkpoint.cpt")
            log_data = pd.DataFrame(
                {
                    "epoch": list(range(epoch + 1)),
                    "train_loss": train_loss_list,
                    "train_recon_pose_loss": train_recon_pose_loss_list,
                    "train_recon_dir_loss": train_recon_dir_loss_list,
                    "train_recon_extra_loss": train_recon_extra_loss_list,
                    "train_vq_loss": train_vq_loss_list,
                    "train_vq_hand_loss": train_vq_hand_loss_list,
                    "train_vq_extra_loss": train_vq_extra_loss_list,
                    "train_perplexity": train_perplexity_list,
                    "eval_loss": eval_loss_list,
                    "eval_recon_pose_loss": eval_recon_pose_loss_list,
                    "eval_recon_dir_loss": eval_recon_dir_loss_list,
                    "eval_recon_extra_loss": eval_recon_extra_loss_list,
                    "eval_recon_vq_loss": eval_recon_vq_loss_list,
                    "eval_recon_vq_hand_loss": eval_recon_vq_hand_loss_list,
                    "eval_recon_vq_extra_loss": eval_recon_vq_extra_loss_list,
                    "eval_recon_perplexity_loss": eval_recon_perplexity_list,
                    "test_loss": test_loss_list,
                    "test_recon_pose_loss": test_recon_pose_loss_list,
                    "test_recon_dir_loss": test_recon_dir_loss_list,
                    "test_recon_extra_loss": test_recon_extra_loss_list,
                    "test_recon_vq_loss": test_recon_vq_loss_list,
                    "test_recon_vq_hand_loss": test_recon_vq_hand_loss_list,
                    "test_recon_vq_extra_loss": test_recon_vq_extra_loss_list,
                    "test_recon_perplexity_loss": test_recon_perplexity_list,
                }
            )
            log_data.to_csv(f"{save_path}/log.csv")
            if self.scheduler is not None and self.config["lr_parameters"]["scheduler_timing"] == "epoch":
                if self.config["lr_parameters"]["scheduler_type"] == "cosinewarmup":
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
    def visualize(self,model,loader,device):
        # 可視化用のディレクトリを作成
        #予測したポーズとGTポーズをそれぞれ256x256の白画像にプロットする
        if os.path.exists(f"{self.config['save_path']}/visualize"):
            shutil.rmtree(f"{self.config['save_path']}/visualize")
        os.makedirs(f"{self.config['save_path']}/visualize",exist_ok=True)
        #予測したポーズとGTポーズを保存するディレクトリを作成
        os.makedirs(f"{self.config['save_path']}/visualize/GT",exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/visualize/Pred",exist_ok=True)

        model.eval()
        prev_hist=None
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader.dataset) // loader.batch_size):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path = batch
                padded_cod_data = padded_cod_data.float().to(device)
                padded_mask = padded_mask.to(device)
                input_length_tensor = input_length_tensor.to(device)
                id_list = id_list.to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                outputs = model.code_usage_histogram(padded_cod_data,prev_hist=prev_hist,hand_valid_mask=padded_mask,input_length=input_length_tensor)
                prev_hist=outputs
                pred_poses=outputs['predicted_poses'].cpu()
                B,T,D=pred_poses.shape
                pred_poses=pred_poses.view(B,T,D//3,3).numpy()
                gt_poses=padded_cod_data[0].cpu()
                gt_poses=gt_poses.permute(0,2,3,1).numpy()
                input_lengths=input_length_tensor.cpu().numpy()
                id_list=id_list.cpu().numpy()
                for i in range(len(id_list)):
                    os.makedirs(f"{self.config['save_path']}/visualize/GT/{data_path[i].split('/')[-1]}",exist_ok=True)
                    os.makedirs(f"{self.config['save_path']}/visualize/Pred/{data_path[i].split('/')[-1]}",exist_ok=True)
                    #opencvを使って白画像にposeをプロット
                    for t in range(input_lengths[i]):
                        gt_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
                        pred_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
                        for f in range(D//3):
                            x_gt=int(gt_poses[i][t][f][0]*256)
                            y_gt=int(gt_poses[i][t][f][1]*256)
                            x_pred=int(pred_poses[i][t][f][0]*256)
                            y_pred=int(pred_poses[i][t][f][1]*256)
                            cv2.circle(gt_img,(x_gt,y_gt),3,(0,0,255),-1)
                            try:
                                cv2.circle(pred_img,(x_pred,y_pred),3,(255,0,0),-1)
                            except:
                                cv2.circle(pred_img, (0, 0), 3, (255, 0, 0), -1)

                        #if stop_logits[i] < t:
                        cv2.imwrite(f"{self.config['save_path']}/visualize/Pred/{data_path[i].split('/')[-1]}/{t:03}.png", pred_img)
                        #if input_lengths[i]>t:
                        cv2.imwrite(f"{self.config['save_path']}/visualize/GT/{data_path[i].split('/')[-1]}/{t:03}.png", gt_img)
            if prev_hist is not None:
                save_code_usage_histogram(prev_hist,f"{self.config['save_path']}/visualize/code_usage_histogram.png",top_k=50,title="Code Usage Histogram")
                save_code_usage_probability(prev_hist,f"{self.config['save_path']}/visualize/code_usage_probability.png",top_k=50,title="Code Usage Probability")

        return

