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




class VQVAETrainer(BaseTrainer):
    def __init__(self,config,scheduler=None):
        self.config=config
        self.scheduler=scheduler
        self.step=0
        self.g_scheduler=0.0
        self.replacement_num_batches=self.config["lr_parameters"]["replacement_num_batches"]
        self.iter=0
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
        total_vq_loss=[]
        total_perplexity=[]
        model.quant.reset_usage()
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
            recon_pose_loss=loss_dict['loss_recon_pos']
            recon_hand_loss_list=loss_dict.get('loss_recon_hand',torch.tensor(0.0))
            loss_vq=loss_dict['loss_vq']
            perplexity=loss_dict['perplexity']
            scaler.scale(loss).backward()
            ##grad_clip
            if self.config["lr_parameters"]["grad_clip_norm"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["lr_parameters"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            total_loss.append(loss.item())
            total_recon_pose_loss.append(recon_pose_loss.item())
            total_recon_hand_loss.append(recon_hand_loss_list.item())
            total_vq_loss.append(loss_vq.item())
            total_perplexity.append(perplexity.item())
            if ((self.iter + 1) % self.replacement_num_batches == 0):
                model.random_restart(loss_dict['z_e'].detach(),threshold=0.5)
            if batch_idx % 100== 0:
                tqdm.write(f"Avg Loss: {np.mean(total_loss)}")
                tqdm.write(f"Avg Recon Pose Loss: {np.mean(total_recon_pose_loss)}")
                tqdm.write(f"Avg Recon Dir Loss: {np.mean(total_recon_hand_loss)}")
                tqdm.write(f"Avg VQ Loss: {np.mean(total_vq_loss)}")
                tqdm.write(f"Avg Perplexity: {np.mean(total_perplexity)}")
                torch.cuda.empty_cache()  # 溜まった断片化を掃除する
            if self.config['lr_parameters']["scheduler_timing"] == "step":
                if self.config["lr_parameters"]["scheduler_type"] == "CosineAnnealingWarmRestarts":
                    self.scheduler.step(self.step)
                else:
                    self.scheduler.step()
                self.step += 1
            self.iter+=1
            del loss_dict,loss,batch



        avg_loss = np.mean(total_loss).astype(np.float32)
        recon_pose_avg_loss=np.mean(total_recon_pose_loss).astype(np.float32)
        recon_hand_avg_loss=np.mean(total_recon_hand_loss).astype(np.float32)
        vq_avg_loss=np.mean(total_vq_loss).astype(np.float32)
        perplexity_avg=np.mean(total_perplexity).astype(np.float32)
        print(model.quant.usage_summary())
        return {
            "loss": avg_loss,
            "recon_pose_loss": recon_pose_avg_loss,
            "recon_hand_loss": recon_hand_avg_loss,
            "vq_loss": vq_avg_loss,
            "perplexity": perplexity_avg
        }
    def eval(self, model, criterion, test_loader, device):
        model.eval()
        total_loss = []
        total_recon_pose_loss=[]
        total_recon_hand_loss=[]
        total_vq_loss=[]
        total_perplexity=[]
        model.quant.reset_usage()

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
                recon_pose_loss = loss_dict['loss_recon_pos']
                recon_hand_loss_list = loss_dict.get('loss_recon_hand', torch.tensor(0.0))
                loss_vq = loss_dict['loss_vq']
                perplexity = loss_dict['perplexity']
                total_loss.append(loss.item())
                total_recon_pose_loss.append(recon_pose_loss.item())
                total_recon_hand_loss.append(recon_hand_loss_list.item())
                total_vq_loss.append(loss_vq.item())
                total_perplexity.append(perplexity.item())
                del loss_dict,loss,batch
        avg_loss = np.mean(total_loss).astype(np.float32)
        recon_pose_avg_loss=np.mean(total_recon_pose_loss).astype(np.float32)
        recon_hand_avg_loss=np.mean(total_recon_hand_loss).astype(np.float32)
        vq_avg_loss=np.mean(total_vq_loss).astype(np.float32)
        perplexity_avg=np.mean(total_perplexity).astype(np.float32)
        print(model.quant.usage_summary())

        return {
            "loss": avg_loss,
            "recon_pose_loss": recon_pose_avg_loss,
            "recon_hand_loss": recon_hand_avg_loss,
            "vq_loss": vq_avg_loss,
            "perplexity": perplexity_avg
        }
    def fit(self,model,optimizer,scheduler,criterion,train_loader,eval_loader,test_loader,device,early_stopping=None):
        if self.config["lr_parameters"]["ema"]:
            ema_model=EMA(model,self.config["lr_parameters"]["ema_beta"])
        num_epochs=self.config["lr_parameters"]['epoch']
        train_loss_list=self.config['train_loss_list'] if 'train_loss_list' in self.config.keys() else []
        train_recon_pose_loss_list=self.config['train_recon_pose_loss_list'] if 'train_recon_pose_loss_list' in self.config.keys() else []
        train_recon_hand_loss_list=self.config['train_recon_hand_loss_list'] if 'train_recon_hand_loss_list' in self.config.keys() else []
        train_vq_loss_list=self.config['train_vq_loss_list'] if 'train_vq_loss_list' in self.config.keys() else []
        train_perplexity_list=self.config['train_perplexity_list'] if 'train_perplexity_list' in self.config.keys() else []
        eval_loss_list=self.config['eval_loss_list'] if 'eval_loss_list' in self.config.keys() else []
        eval_recon_pose_loss_list=self.config['eval_recon_pose_loss_list'] if 'eval_pose_loss_list' in self.config.keys() else []
        eval_recon_hand_loss_list=self.config['eval_recon_hand_loss_list'] if 'eval_recon_hand_loss_list' in self.config.keys() else []
        eval_recon_vq_loss_list=self.config['eval_recon_vq_loss_list'] if 'eval_recon_vq_loss_list' in self.config.keys() else []
        eval_recon_perplexity_list=self.config['eval_recon_perplexity_list'] if 'eval_recon_perplexity_list' in self.config.keys() else []
        test_loss_list=self.config['test_loss_list'] if 'test_loss_list' in self.config.keys() else []
        test_recon_pose_loss_list=self.config['test_recon_pose_loss_list'] if 'test_recon_pose_loss_list' in self.config.keys() else []
        test_recon_hand_loss_list=self.config['test_recon_hand_loss_list'] if 'test_recon_hand_loss_list' in self.config.keys() else []
        test_recon_vq_loss_list=self.config['test_recon_vq_loss_list'] if 'test_recon_vq_loss_list' in self.config.keys() else []
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
            train_recon_hand_loss_list.append(train_loss['recon_hand_loss'])
            train_vq_loss_list.append(train_loss['vq_loss'])
            train_perplexity_list.append(train_loss['perplexity'])
            eval_loss_list.append(eval_loss['loss'])
            eval_recon_pose_loss_list.append(eval_loss['recon_pose_loss'])
            eval_recon_hand_loss_list.append(eval_loss['recon_hand_loss'])
            eval_recon_vq_loss_list.append(eval_loss['vq_loss'])
            eval_recon_perplexity_list.append(eval_loss['perplexity'])
            test_loss_list.append(test_loss['loss'])
            test_recon_pose_loss_list.append(test_loss['recon_pose_loss'])
            test_recon_hand_loss_list.append(test_loss['recon_hand_loss'])
            test_recon_vq_loss_list.append(test_loss['vq_loss'])
            test_recon_perplexity_list.append(test_loss['perplexity'])
            print(f"Epoch {epoch+1}/{num_epochs})")
            print(f"Train Loss: {train_loss['loss']:.4f}, Recon Pose Loss: {train_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {train_loss['recon_hand_loss']:.4f},  VQ Loss: {train_loss['vq_loss']:.4f}, Perplexity: {train_loss['perplexity']:.4f}")
            print(f"Eval Loss: {eval_loss['loss']:.4f}, Recon Pose Loss: {eval_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {eval_loss['recon_hand_loss']:.4f},  VQ Loss: {eval_loss['vq_loss']:.4f}, Perplexity: {eval_loss['perplexity']:.4f}")
            print(f"Test Loss: {test_loss['loss']:.4f}, Recon Pose Loss: {test_loss['recon_pose_loss']:.4f}, Recon Hand Loss: {test_loss['recon_hand_loss']:.4f}, VQ Loss: {test_loss['vq_loss']:.4f}, Perplexity: {test_loss['perplexity']:.4f}")

            #eval_lossとtest_lossのkeyを変更
            eval_loss = {
                "eval_loss": eval_loss['loss'],
                "eval_recon_pose_loss": eval_loss['recon_pose_loss'],
                "eval_recon_hand_loss": eval_loss['recon_hand_loss'],
                "eval_vq_loss": eval_loss['vq_loss'],
                "eval_perplexity": eval_loss['perplexity']
            }
            test_loss = {
                "test_loss": test_loss['loss'],
                "test_recon_pose_loss": test_loss['recon_pose_loss'],
                "test_recon_hand_loss": test_loss['recon_hand_loss'],
                "test_vq_loss": test_loss['vq_loss'],
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
                "train_recon_hand_loss_list": train_recon_hand_loss_list,
                "train_vq_loss_list": train_vq_loss_list,
                "train_perplexity_list": train_perplexity_list,
                "eval_loss_list": eval_loss_list,
                "eval_recon_pose_loss_list": eval_recon_pose_loss_list,
                "eval_recon_hand_loss_list": eval_recon_hand_loss_list,
                "eval_recon_vq_loss_list": eval_recon_vq_loss_list,
                "eval_recon_perplexity_list": eval_recon_perplexity_list,
                "test_loss_list": test_loss_list,
                "test_recon_pose_loss_list": test_recon_pose_loss_list,
                "test_recon_hand_loss_list": test_recon_hand_loss_list,
                "test_recon_vq_loss_list": test_recon_vq_loss_list,
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
                    "train_recon_hand_loss": train_recon_hand_loss_list,
                    "train_vq_loss": train_vq_loss_list,
                    "train_perplexity": train_perplexity_list,
                    "eval_loss": eval_loss_list,
                    "eval_recon_pose_loss": eval_recon_pose_loss_list,
                    "eval_recon_hand_loss": eval_recon_hand_loss_list,
                    "eval_recon_vq_loss": eval_recon_vq_loss_list,
                    "eval_recon_perplexity_loss": eval_recon_perplexity_list,
                    "test_loss": test_loss_list,
                    "test_recon_pose_loss": test_recon_pose_loss_list,
                    "test_recon_hand_loss": test_recon_hand_loss_list,
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
    def visualize(self,model,dataset,device):
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
        dataset.set_return_length()

        with torch.no_grad():
            for batch in tqdm(dataset, total=len(dataset)):
                padded_cod_data, padded_mask, input_length_tensor, id_list, data_path, center_data, shoulder_length, left_center_data, left_length, right_center_data, right_length = batch
                padded_cod_data = padded_cod_data.float().unsqueeze(0).to(device)
                padded_mask = padded_mask.unsqueeze(0).to(device)
                input_length_tensor = input_length_tensor.unsqueeze(0).to(device)
                id_list = torch.tensor(id_list).to(device)
                batch = (padded_cod_data, padded_mask, input_length_tensor, id_list)
                model_output = model(padded_cod_data,hand_valid_mask=padded_mask,input_length=input_length_tensor)
                output=model_output['output']
                codes=model_output['codes']
                prev_hist = model.code_usage_histogram_update(padded_cod_data,prev_hist,input_length=input_length_tensor)
                output = output.squeeze(0).cpu().numpy()
                T, J, C = output.shape
                # outputを元のスケールに戻す
                output = output.reshape(T, J, C)
                output[:, :8] *= shoulder_length.cpu().numpy()[:, None, None]
                output[:, :8] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                pd_left_center_data=output[:,6]
                pd_right_center_data=output[:,7]
                output[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 8:29] += pd_left_center_data[:, None, :]
                output[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                output[:, 29:] += pd_right_center_data[:, None, :]
                # output=average_movint(output.reshape(T,J*C),window_size=7).reshape(T,J,C)
                output = np.where(output == 0, np.nan, output)
                output = apply_savgol_filter(output.transpose(1, 0, 2), window_size=7, poly_order=2).transpose(1, 0, 2)
                output = np.where(np.isnan(output), 0, output)
                # output[:,8:29]で，全てが0.01以下のときは，手が検出されなかったとして，0にする
                hand_mask = (np.max(np.abs(output[:, 8:29]), axis=2) < 0.01)
                output[:, 8:29][hand_mask] = 0.0
                # output[:,29:]も同様
                hand_mask = (np.max(np.abs(output[:, 29:]), axis=2) < 0.01)
                output[:, 29:][hand_mask] = 0.0

                padded_cod_data = padded_cod_data.squeeze(0).cpu().numpy()
                padded_cod_data = padded_cod_data.reshape(T, J, C)
                padded_cod_data[:, :8] *= shoulder_length.cpu().numpy()[:, None, None]
                padded_cod_data[:, :8] += center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 8:29] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 8:29] += left_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                padded_cod_data[:, 29:] *= shoulder_length.cpu().numpy()[:, None, None] / 2
                padded_cod_data[:, 29:] += right_center_data.cpu().transpose(0, 1).numpy()[:, None, :]
                # outputの点群を動画として保存
                # 同時に，元の動画も保存
                video_size = (256, 256)
                v_writer = cv2.VideoWriter(
                    f"{self.config['save_path']}/visualize/Pred/pred_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                v_writer_gt = cv2.VideoWriter(
                    f"{self.config['save_path']}/visualize/GT/gt_{os.path.basename(data_path)}.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, video_size)
                for t in range(T):
                    base_frame = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    base_frame_gt = np.ones((video_size[0], video_size[1], 3), dtype=np.uint8) * 255
                    for j in range(J):
                        x = int(output[t, j, 0] * video_size[0])
                        y = int(output[t, j, 1] * video_size[1])
                        x_gt = int(padded_cod_data[t, j, 0] * video_size[0])
                        y_gt = int(padded_cod_data[t, j, 1] * video_size[1])
                        pd_frame = cv2.circle(base_frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                        gt_frame = cv2.circle(base_frame_gt, (x_gt, y_gt), radius=3, color=(0, 255, 0), thickness=-1)
                    v_writer.write(pd_frame)
                    v_writer_gt.write(gt_frame)
                v_writer.release()
                v_writer_gt.release()
            if prev_hist is not None:
                save_code_usage_histogram(prev_hist, f"{self.config['save_path']}/visualize/code_usage_histogram.png", top_k=10, title=f"Code Usage Histogram")
                save_code_usage_probability(prev_hist, f"{self.config['save_path']}/visualize/code_usage_probability.png", top_k=10, title=f"Code Usage Probability ")
            return

