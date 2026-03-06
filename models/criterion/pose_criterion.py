import torch
import torch.nn as nn
from loader.coordinate_preprocess import all_connections,connection_to_set,connections,hand_connections
import torch.nn.functional as F
# List of connections between landmarks (0 to 20)
# List of connections between landmarks (0 to 20)

def sort_connections(connections):
    #all_connectionsは元データのインデックスを表すが，データからconnectionに対応するデータを取得すると，インデックスが変わる
    #all_connectionsのインデックスを変換後のデータに対応させる
    #all_connectionsにあるインデックスを重複無しで抽出
    unique_indices = []
    for conn in connections:
        for idx in conn:
            if idx not in unique_indices:
                unique_indices.append(idx)
    #print(unique_indices)
    index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(unique_indices)}
    #print(index_mapping)
    #all_connectionsのインデックスを変換
    sorted_connections = []
    for conn in connections:
        sorted_conn = (index_mapping[conn[0]], index_mapping[conn[1]])
        sorted_connections.append(sorted_conn)
    return sorted_connections
class Text2Pose_criterion(nn.Module):
    def __init__(self,config):
        super(Text2Pose_criterion, self).__init__()
        self.config=config
        self.pose_loss_weight=config['pose_loss_weight']
        self.bone_loss_weight=config['bone_loss_weight']
        self.pose_loss=nn.L1Loss(reduction='none')
        self.stop_loss=nn.BCEWithLogitsLoss(reduction='none')
        self.bone_loss=nn.MSELoss(reduction='none')
    def bone_criterion(self, predicted_poses, target_poses, connections):
        #predicted_poses,target_poses:(batch_size, seq_len, pose_dim)
        #connections:骨格の接続リスト
        #2転換のboneのベクトルの2乗誤差を計算
        batch_size, seq_len, pose_dim = predicted_poses.size()
        predicted_poses=predicted_poses.reshape(batch_size, seq_len, pose_dim//2,2)
        target_poses=target_poses.reshape(batch_size, seq_len, pose_dim//2,2)
        bone_loss=torch.zeros((batch_size,seq_len),dtype=torch.float32,device=predicted_poses.device)
        connections=sort_connections(connections)
        for conn in connections:
            bone_start_pred=predicted_poses[:,:,conn[0],:]  #(batch_size, seq_len, 2)
            bone_end_pred=predicted_poses[:,:,conn[1],:]    #(batch_size, seq_len, 2)
            bone_start_target=target_poses[:,:,conn[0],:]   #(batch_size, seq_len, 2)
            bone_end_target=target_poses[:,:,conn[1],:]     #(batch_size, seq_len, 2)
            bone_vec_pred=bone_end_pred - bone_start_pred  #(batch_size, seq_len, 2)
            bone_vec_target=bone_end_target - bone_start_target #(batch_size, seq_len, 2)
            bone_loss+=self.bone_loss(bone_vec_pred,bone_vec_target).mean(dim=-1)  #(batch_size, seq_len)
        bone_loss=bone_loss/(len(connections)+1e-8)
        return bone_loss #(batch_size, seq_len)
    def velocity_criterion(self, predicted_poses, target_poses):
        #predicted_poses,target_poses:(batch_size, seq_len, pose_dim)
        #2転換のvelocityのベクトルの2乗誤差を計算
        batch_size, seq_len, pose_dim = predicted_poses.size()
        predicted_velocity=predicted_poses[:,1:,:]-predicted_poses[:,:-1,:] #(batch_size, seq_len-1, pose_dim)
        target_velocity=target_poses[:,1:,:]-target_poses[:,:-1,:] #(batch_size, seq_len-1, pose_dim)
        velocity_loss=self.pose_loss(predicted_velocity,target_velocity).mean(dim=-1) #(batch_size, seq_len-1)
        return velocity_loss #(batch_size, seq_len-1)
    def create_stop_targets(self, target_poses, target_length):
        #target_poses: (batch_size, seq_len, pose_dim)
        #target_length: (batch_size,)
        batch_size, seq_len, _ = target_poses.size()
        stop_targets = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=target_poses.device)
        for i in range(batch_size):
            stop_targets[i, target_length[i]-1] = 1.0  # シーケンスの最後のフレームで停止
        return stop_targets  #(batch_size, seq_len)
    def create_mask(self, target_length, max_len):
        #target_length: (batch_size,)
        batch_size = target_length.size(0)
        mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=target_length.device)
        for i in range(batch_size):
            mask[i, :target_length[i]] = 1.0
        return mask  #(batch_size, max_len)
    def forward(self,model_output, target_poses, target_length):
        predicted_poses=model_output["predicted_poses"]

        #stop_targetsの作成
        target_poses=target_poses.permute(0,2,3,1)
        target_poses=target_poses.reshape(target_poses.size(0),target_poses.size(1),-1)
        #ポーズ再構成損失の計算
        pose_loss=self.pose_loss(predicted_poses[:,:-1], target_poses[:,1:])
        #シーケンス長に基づいてマスクを作成
        max_len=predicted_poses.size(1)
        mask=self.create_mask(target_length,max_len)
        pose_mask=self.create_mask(target_length-1,max_len-1)
        pose_loss=(pose_loss.mean(dim=-1)*pose_mask).sum()/pose_mask.sum()
        #停止損失の計算
        #bone lossの計算(骨格の自然さを促進)
        bone_loss=self.bone_criterion(predicted_poses, target_poses, all_connections)#(batch_size, seq_len)
        bone_loss=(bone_loss*mask).sum()/mask.sum()
        #pose_loss+=bone_loss
        #総損失の計算
        total_loss=self.pose_loss_weight*pose_loss+self.bone_loss_weight*bone_loss
        return {
            "loss": total_loss,
            "pose_loss": pose_loss,
            "bone_loss": bone_loss
        }
    def generate_forward(self,model_output, target_poses, target_length,timestep):
        predicted_poses = model_output["predicted_poses"]
        target_poses = target_poses.permute(0, 2, 3, 1)
        target_poses = target_poses.reshape(target_poses.size(0), target_poses.size(1), -1)
        # ポーズ再構成損失の計算
        pose_loss = self.pose_loss(predicted_poses[:,timestep], target_poses[:,timestep])#(B, pose_dim)
        # シーケンス長に基づいてマスクを作成
        max_len = predicted_poses.size(1)
        mask = self.create_mask(target_length, max_len)[:, timestep]#(B,)
        pose_loss = (pose_loss.mean(dim=-1) * mask).sum() / mask.sum()
        # bone lossの計算(骨格の自然さを促進)
        bone_loss = self.bone_criterion(predicted_poses[:,timestep].unsqueeze(1), target_poses[:,timestep].unsqueeze(1), all_connections).squeeze(1)  # (batch_size, seq_len)
        bone_loss = (bone_loss * mask).sum() / mask.sum()
        total_loss=self.pose_loss_weight*pose_loss+self.bone_loss_weight*bone_loss

        return {
            "loss": total_loss,
            "pose_loss": pose_loss,
            "bone_loss": bone_loss
        }
class Text2PoseDiffusion_criterion(Text2Pose_criterion):
    def __init__(self,config):
        super(Text2PoseDiffusion_criterion, self).__init__(config)
        self.pose_loss=nn.L1Loss(reduction='none')


    def forward(self, model_output, t_poses, target_length):
        predicted_poses = model_output["predicted_poses"]
        if 'pose_length' in model_output:
            target_length=model_output['pose_length']
        if "v_targets" in model_output:
            target_poses=model_output["v_targets"]
        elif "eps_targets" in model_output:
            target_poses=model_output["eps_targets"]
        else:
            # stop_targetsの作成
            if 'target_poses' in model_output:
                target_poses=model_output['target_poses']
            else:
                target_poses = t_poses.permute(0, 2, 3, 1)
                target_poses = target_poses.reshape(target_poses.size(0), target_poses.size(1), -1)
        # ポーズ再構成損失の計算
        body_poses=predicted_poses[:,:,:18]
        target_body_poses=target_poses[:,:,:18]
        hand_poses=predicted_poses[:,:,18:]
        target_hand_poses=target_poses[:,:,18:]
        body_pose_loss=self.pose_loss(body_poses, target_body_poses)
        hand_pose_loss=self.pose_loss(hand_poses, target_hand_poses)
        pose_loss = self.config['pose_weights'] * body_pose_loss.mean(dim=-1) + self.config['hand_weights'] * hand_pose_loss.mean(dim=-1)
        #pose_loss=self.pose_loss(predicted_poses, target_poses).mean(dim=-1)
        # シーケンス長に基づいてマスクを作成
        max_len = predicted_poses.size(1)
        mask = self.create_mask(target_length, max_len)
        pose_loss = (pose_loss * mask).sum() / mask.sum()
        # bone lossの計算(骨格の自然さを促進)
        if "v_targets" in model_output or "eps_targets" in model_output:
            predicted_poses=model_output["x0_pred"]
            target_poses = t_poses.permute(0, 2, 3, 1)
            target_poses = target_poses.reshape(target_poses.size(0), target_poses.size(1), -1)
        bone_loss = self.bone_criterion(predicted_poses, target_poses, all_connections)  # (batch_size, seq_len)
        bone_loss = (bone_loss * mask).sum() / mask.sum()
        #velocity_lossの計算(動きの自然さを促進)
        velocity_bone_loss=self.velocity_criterion(body_poses, target_body_poses)
        velocity_hand_loss=self.velocity_criterion(hand_poses, target_hand_poses)
        velocity_loss=self.config['pose_weights']*velocity_bone_loss+self.config['hand_weights']*velocity_hand_loss
        vel_mask=self.create_mask(target_length-1,max_len-1)
        velocity_loss=(velocity_loss*vel_mask).sum()/vel_mask.sum()
        # pose_loss+=bone_loss
        # 総損失の計算
        total_loss = self.pose_loss_weight * pose_loss + self.bone_loss_weight * bone_loss+ self.config['velocity_loss_weight']*velocity_loss
        return {
            "loss": total_loss,
            "pose_loss": pose_loss,
            "bone_loss": velocity_loss
        }
