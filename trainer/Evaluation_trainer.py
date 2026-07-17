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

class EvaluationTrainer:
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
    def train(self, model, optimizer, criterion, train_loader, device,ema=False):
        model.train()
        total_loss = []
        total_accuracy = 0
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["lr_parameters"]["amp"])
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader.dataset) // train_loader.batch_size):
            padded_cod_data,padded_mask, input_length_tensor, id_list,data_path,sequence=batch
            padded_cod_data=padded_cod_data.float().to(device)
            padded_mask=padded_mask.to(device)
            input_length_tensor=input_length_tensor.to(device)
            id_list=id_list.to(device)
            sequence=sequence.to(device)
            optimizer.zero_grad(set_to_none=True)
            #g_prob = random.random()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                output = model(padded_cod_data,input_length_tensor,sequence)


            loss = output['loss']
            prob=F.softmax(output['logits'])
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
            if self.config['lr_parameters']["scheduler_timing"] == "step":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(self.step)
                else:
                    self.scheduler.step()
                self.step+=1

            # codebook replacement
            if batch_idx % 100== 0:
                tqdm.write(f"Avg Loss: {np.mean(total_loss)}")
            #正解数
            acc=(prob.argmax(dim=1) == sequence).sum().item()
            total_accuracy+=acc



        avg_loss = np.mean(total_loss).astype(np.float32)
        avg_acc= total_accuracy / len(train_loader.dataset)
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
    def eval(self, model, criterion, test_loader, device):
        model.eval()
        total_loss = []
        total_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader.dataset) // test_loader.batch_size):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, sequence = batch
                padded_cod_data = padded_cod_data.float().to(device)
                padded_mask = padded_mask.to(device)
                input_length_tensor = input_length_tensor.to(device)
                id_list = id_list.to(device)
                sequence=sequence.to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list, sequence)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.config["lr_parameters"]["amp"]):
                    output = model(padded_cod_data, input_length_tensor, sequence)

                loss = output['loss']
                logits=F.softmax(output['logits'],dim=-1)
                total_loss.append(loss.item())
                acc=(logits.argmax(dim=1) == sequence).sum().item()
                total_accuracy+=acc
        avg_loss = np.mean(total_loss).astype(np.float32)
        avg_acc = total_accuracy / len(test_loader.dataset)

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
    def fit(self,model,optimizer,scheduler,criterion,train_loader,eval_loader,test_loader,device,early_stopping=None):
        if self.config["lr_parameters"]["ema"]:
            ema_model=EMA(model,self.config["lr_parameters"]["ema_beta"])
        num_epochs=self.config["lr_parameters"]['epoch']
        train_loss_list=self.config['train_loss_list'] if 'train_loss_list' in self.config.keys() else []
        train_acc_list=self.config['train_acc_list'] if 'train_acc_list' in self.config.keys() else []
        eval_loss_list=self.config['eval_loss_list'] if 'eval_loss_list' in self.config.keys() else []
        eval_acc_list=self.config['eval_acc_list'] if 'eval_acc_list' in self.config.keys() else []
        test_loss_list=self.config['test_loss_list'] if 'test_loss_list' in self.config.keys() else []
        test_acc_list=self.config['test_acc_list'] if 'test_acc_list' in self.config.keys() else []

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
            train_acc_list.append(train_loss['accuracy'])

            eval_loss_list.append(eval_loss['loss'])
            eval_acc_list.append(eval_loss['accuracy'])

            test_loss_list.append(test_loss['loss'])
            test_acc_list.append(test_loss['accuracy'])

            print(f"Epoch {epoch+1}/{num_epochs})")
            print(f"Train Loss: {train_loss['loss']:.4f}, Train Acc: {train_loss['accuracy']:.4f}")
            print(f"Eval Loss: {eval_loss['loss']:.4f}, Eval Acc: {eval_loss['accuracy']:.4f}")
            print(f"Test Loss: {test_loss['loss']:.4f}, Test Acc: {test_loss['accuracy']:.4f}")


            #eval_lossとtest_lossのkeyを変更
            eval_loss = {
                "eval_loss": eval_loss['loss'],
                'eval_accuracy': eval_loss['accuracy'],
            }
            test_loss = {
                "test_loss": test_loss['loss'],
                "test_accuracy": test_loss['accuracy'],
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
                "train_acc_list": train_acc_list,
                "eval_loss_list": eval_loss_list,
                "eval_acc_list": eval_acc_list,
                "test_loss_list": test_loss_list,
                "test_acc_list": test_acc_list,
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
                    "train_accuracy": train_acc_list,
                    "eval_loss": eval_loss_list,
                    "eval_accuracy": eval_acc_list,
                    "test_loss": test_loss_list,
                    "test_accuracy": test_acc_list,
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

    def evaluation(self, model,vae_model, dataset, device,visualize_dir_name="visualize"):
        # model: eval用のモデル
        # vae_model: ポーズ生成用のモデル
        # dataset: 可視化に使用するデータセット
        # device: 使用するデバイス
        model.eval()
        vae_model.eval()
        dataset.set_return_length()
        eval_acc=0
        gt_acc=0
        l2_metric=0
        with torch.no_grad():
            for batch in tqdm(dataset, total=len(dataset)):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path,sequence,n_class, center_data, shoulder_length, left_center_data, left_length, right_center_data, right_length = batch
                padded_cod_data = padded_cod_data.float().unsqueeze(0).to(device)
                B,T,J,C=padded_cod_data.shape
                padded_mask = padded_mask.unsqueeze(0).to(device)
                input_length_tensor = input_length_tensor.unsqueeze(0).to(device)
                id_list = torch.tensor(id_list).to(device)
                sequence=sequence.to(device)
                n_class=torch.tensor(n_class).unsqueeze(0).to(device)
                output=vae_model(padded_cod_data, input_length_tensor, sequence)['text_output'].reshape(B,T,J,C)
                eval_output = model(output, input_length_tensor, n_class)
                gt_output=model(padded_cod_data, input_length_tensor, n_class)
                eval_logits=F.softmax(eval_output['logits'],dim=-1)
                gt_logits=F.softmax(gt_output['logits'],dim=-1)
                eval_acc+=(eval_logits.argmax(dim=1) == n_class).sum().item()
                gt_acc+=(gt_logits.argmax(dim=1) == n_class).sum().item()
                output=output.cpu()
                padded_cod_data=padded_cod_data.cpu()
                output=output.reshape(T,J,C)
                output[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 8:29] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                output[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 29:] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]

                padded_cod_data = padded_cod_data.cpu()
                padded_cod_data = padded_cod_data.reshape(T, J, C)
                padded_cod_data[:, :8] *= shoulder_length.cpu().numpy()[:, None, None]
                padded_cod_data[:, :8] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 8:29] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 29:] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]

                l2_metric+=torch.sqrt(F.mse_loss(output, padded_cod_data)).item()
        eval_accuracy=eval_acc/len(dataset)
        gt_accuracy=gt_acc/len(dataset)
        l2_metric=l2_metric/len(dataset)
        print(f"Evaluation Accuracy: {eval_accuracy:.4f}")
        print(f"GT Accuracy: {gt_accuracy:.4f}")
        print(f"L2 Metric: {l2_metric:.4f}")
        #結果を保存(csv)
        results_df=pd.DataFrame({
            "eval_accuracy": [eval_accuracy],
            "gt_accuracy": [gt_accuracy],
            "l2_metric": [l2_metric],
        })
        os.makedirs(visualize_dir_name, exist_ok=True)
        results_df.to_csv(f"{self.config['save_path']}/{visualize_dir_name}.csv", index=False)

        return




