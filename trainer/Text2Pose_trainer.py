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


class Text2PoseTrainer(BaseTrainer):
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
        padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list = batch

        outputs = model(padded_tokens_tensor,id_list,padded_cod_data[0],input_length_tensor)
        loss = criterion(outputs, padded_cod_data[0],input_length_tensor)
        return loss
    def train(self, model, optimizer, criterion, train_loader, device,ema=False):
        model.train()
        total_loss = []
        total_pose_loss=[]
        total_bone_loss=[]
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["lr_parameters"]["amp"])
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader.dataset) // train_loader.batch_size):
            padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list,data_path=batch
            padded_cod_data=[item.float().to(device) for item in padded_cod_data]
            input_length_tensor=input_length_tensor.to(device)
            padded_tokens_tensor=padded_tokens_tensor.to(device)
            id_list=id_list.to(device)
            batch = (padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list)
            optimizer.zero_grad(set_to_none=True)
            #g_prob = random.random()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                loss_dict = self.compute_loss(batch, model, criterion)

            loss = loss_dict['loss']
            pose_loss=loss_dict['pose_loss']
            bone_loss_list=loss_dict['bone_loss']
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
            total_pose_loss.append(pose_loss.item())
            total_bone_loss.append(bone_loss_list.item())
            if batch_idx % 100== 0:
                tqdm.write(f"Avg Loss: {np.mean(total_loss)}")
                tqdm.write(f"Avg Pose Loss: {np.mean(total_pose_loss)}")
                tqdm.write(f"Avg Joint Loss: {np.mean(total_bone_loss)}")



        avg_loss = np.mean(total_loss).astype(np.float32)
        pose_avg_loss=np.mean(total_pose_loss).astype(np.float32)
        bone_avg_loss=np.mean(total_bone_loss).astype(np.float32)
        return {
            "loss": avg_loss,
            "pose_loss": pose_avg_loss,
            "bone_loss": bone_avg_loss
        }
    def eval(self, model, criterion, test_loader, device):
        model.eval()
        total_loss = []
        total_pose_loss=[]
        total_bone_loss=[]
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader.dataset) // test_loader.batch_size):
                padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list,data_path = batch
                padded_cod_data = [item.float().to(device) for item in padded_cod_data]
                input_length_tensor = input_length_tensor.to(device)
                padded_tokens_tensor = padded_tokens_tensor.to(device)
                id_list = id_list.to(device)
                batch = (padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16,enabled=self.config["lr_parameters"]["amp"]):
                    loss_dict = self.compute_loss(batch, model, criterion)
                loss = loss_dict['loss']
                pose_loss=loss_dict['pose_loss']
                bone_loss=loss_dict['bone_loss']
                total_loss.append(loss.item())
                total_pose_loss.append(pose_loss.item())
                total_bone_loss.append(bone_loss.item())
        avg_loss = np.mean(total_loss).astype(np.float32)
        pose_avg_loss=np.mean(total_pose_loss).astype(np.float32)
        bone_avg_loss=np.mean(total_bone_loss).astype(np.float32)
        return {
            "loss": avg_loss,
            "pose_loss": pose_avg_loss,
            "bone_loss": bone_avg_loss
        }
    def fit(self,model,optimizer,scheduler,criterion,train_loader,eval_loader,test_loader,device,early_stopping=None):
        if self.config["lr_parameters"]["ema"]:
            ema_model=EMA(model,self.config["lr_parameters"]["ema_beta"])
        num_epochs=self.config["lr_parameters"]['epoch']
        train_loss_list=self.config['train_loss_list'] if 'train_loss_list' in self.config.keys() else []
        train_pose_loss_list=self.config['train_pose_loss_list'] if 'train_pose_loss_list' in self.config.keys() else []
        train_bone_loss_list=self.config['train_bone_loss_list'] if 'train_bone_loss_list' in self.config.keys() else []
        eval_loss_list=self.config['eval_loss_list'] if 'eval_loss_list' in self.config.keys() else []
        eval_pose_loss_list=self.config['eval_pose_loss_list'] if 'eval_pose_loss_list' in self.config.keys() else []
        eval_bone_loss_list=self.config['eval_bone_loss_list'] if 'eval_bone_loss_list' in self.config.keys() else []
        test_loss_list=self.config['test_loss_list'] if 'test_loss_list' in self.config.keys() else []
        test_pose_loss_list=self.config['test_pose_loss_list'] if 'test_pose_loss_list' in self.config.keys() else []
        test_bone_loss_list=self.config['test_bone_loss_list'] if 'test_bone_loss_list' in self.config.keys() else []
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
            train_pose_loss_list.append(train_loss['pose_loss'])
            train_bone_loss_list.append(train_loss['bone_loss'])
            eval_loss_list.append(eval_loss['loss'])
            eval_pose_loss_list.append(eval_loss['pose_loss'])
            eval_bone_loss_list.append(eval_loss['bone_loss'])
            test_loss_list.append(test_loss['loss'])
            test_pose_loss_list.append(test_loss['pose_loss'])
            test_bone_loss_list.append(test_loss['bone_loss'])
            print(f"Epoch {epoch+1}/{num_epochs})")
            print(f"Train Loss: {train_loss['loss']},Train Pose Loss: {train_loss['pose_loss']}, Train Bone Loss: {train_loss['bone_loss']}")
            print(f"Eval Loss: {eval_loss['loss']},Eval Pose Loss: {eval_loss['pose_loss']},Eval Bone Loss: {eval_loss['bone_loss']}")
            print(f"Test Loss: {test_loss['loss']},Test Pose Loss: {test_loss['pose_loss']}, Test Bone Loss: {test_loss['bone_loss']}")
            #eval_lossとtest_lossのkeyを変更
            eval_loss = {
                "eval_loss": eval_loss['loss'],
                "eval_pose_loss": eval_loss['pose_loss'],
                "eval_bone_loss": eval_loss['bone_loss']
            }
            test_loss = {
                "test_loss": test_loss['loss'],
                "test_pose_loss": test_loss['pose_loss'],
                "test_bone_loss": test_loss['bone_loss']
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
                "train_pose_loss_list": train_pose_loss_list,
                "train_bone_loss_list": train_bone_loss_list,
                "eval_loss_list": eval_loss_list,
                "eval_mse_loss_list": eval_pose_loss_list,
                "eval_bone_loss_list": eval_bone_loss_list,
                "test_loss_list": test_loss_list,
                "test_pose_loss_list": test_pose_loss_list,
                "test_bone_loss_list": test_bone_loss_list,
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
                    "train_pose_loss": train_pose_loss_list,
                    "train_bone_loss": train_bone_loss_list,
                    "eval_loss": eval_loss_list,
                    "eval_pose_loss": eval_pose_loss_list,
                    "eval_bone_loss": eval_bone_loss_list,
                    "test_loss": test_loss_list,
                    "test_pose_loss": test_pose_loss_list,
                    "test_bone_loss": test_bone_loss_list,
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
    def visualize(self,model,loader):
        # 可視化用のディレクトリを作成
        #予測したポーズとGTポーズをそれぞれ256x256の白画像にプロットする
        if os.path.exists(f"{self.config['save_path']}/visualize"):
            shutil.rmtree(f"{self.config['save_path']}/visualize")
        os.makedirs(f"{self.config['save_path']}/visualize",exist_ok=True)
        #予測したポーズとGTポーズを保存するディレクトリを作成
        os.makedirs(f"{self.config['save_path']}/visualize/GT",exist_ok=True)
        os.makedirs(f"{self.config['save_path']}/visualize/Pred",exist_ok=True)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader.dataset) // loader.batch_size):
                padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list,data_path = batch
                padded_cod_data = [item.float().to(self.config['device']) for item in padded_cod_data]
                input_length_tensor = input_length_tensor.to(self.config['device'])
                padded_tokens_tensor = padded_tokens_tensor.to(self.config['device'])
                id_list = id_list.to(self.config['device'])
                batch = (padded_cod_data, input_length_tensor, padded_tokens_tensor, id_list)
                outputs = model.generate(padded_tokens_tensor,id_list,padded_cod_data[0],input_length_tensor)
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

        return

