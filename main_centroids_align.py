from loader.spectrul_clustering import *
from loader.data_loader import *
from loader.coordinate_preprocess import *
import os,glob
from Parameter.Parameter import *
from tslearn.utils import to_time_series_dataset
import pickle,shutil
def diagnose_affinity(K):
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    print(f"K非対角: mean={off.mean():.3f} std={off.std():.3f} "
          f"min={off.min():.3f} max={off.max():.3f}  (std/mean={off.std()/off.mean():.3f})")
    d = K.sum(1); dis = 1/np.sqrt(np.clip(d, 1e-12, None))
    L = np.eye(n) - dis[:, None]*K*dis[None, :]
    ev = np.sort(np.linalg.eigvalsh(L))[:12]
    print("最小固有値:", np.round(ev, 4))
    gaps = np.diff(ev)[1:]
    print(f"最大eigengap: k={int(gaps.argmax()+2)} gap={gaps.max():.4f}")
def gak_outlier_scores(K, k=10):
    n = K.shape[0]
    Koff = K.copy(); np.fill_diagonal(Koff, -np.inf)
    # (a) k近傍平均類似度: 小さいほど外れ
    knn_sim = np.sort(Koff, axis=1)[:, -k:].mean(1)
    # (b) 全体平均類似度: 集団中心からの遠さ
    mean_sim = (K.sum(1) - 1.0) / (n - 1)
    return knn_sim, mean_sim
def kernel_distance_to_center(K):
    n = K.shape[0]
    cross  = K.sum(1) / n                  # ⟨φ(x_i), m⟩
    normsq = K.sum() / n**2                # ‖m‖²（全サンプル共通の定数）
    dist2  = np.diag(K) - 2*cross + normsq
    return dist2                            # 大きいほど集団中心から遠い＝外れ
def robust_flags(score, z=3.5):
    med = np.median(score); mad = np.median(np.abs(score - med)) + 1e-8
    return (score - med) / (1.4826 * mad) > z     # True が外れ値
def sign_video_feature_extractoion(dataset,save_path):

    # opencvでフレーム数を取得
    if dataset == "CSL-Daily":
        ext = "jpg"
    else:
        ext = "png"
    train_path, dev_path, test_path, gloss2class, class2gloss, video2gloss = islr_datasets_loader(dataset)
    if dataset=="phoenixT":
        train_cod_root=f"{WORDS_DATADIR_T_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_T_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_T_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_T_SKELETON_FACE}/test"
    elif dataset=="CSL-Daily":
        train_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/train"
        train_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/train"
        dev_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/dev"
        dev_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/dev"
        test_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON}/test"
        test_face_cod_root=f"{WORDS_DATADIR_CSL_DAILY_SKELETON_FACE}/test"
    elif dataset=="AUTSL":
        train_cod_root=SKELETON_AUTSL_TRAIN_DATADIR_3D
        train_face_cod_root=FACE_AUTSL_TRAIN_DATADIR_3D
        dev_cod_root=SKELETON_AUTSL_DEV_DATADIR_3D
        dev_face_cod_root=FACE_AUTSL_DEV_DATADIR_3D
        test_cod_root=SKELETON_AUTSL_TEST_DATADIR_3D
        test_face_cod_root=FACE_AUTSL_TEST_DATADIR_3D
    else:
        raise ValueError("Invalid dataset name")
    if not os.path.exists(save_path):
        #shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
    centroids_dict={}
    gloss_data_dict={}
    c=0
    for data_path in train_path:
        print(f"Processing {data_path}...")
        gloss_dir=os.path.basename(os.path.dirname(data_path))
        file_name=data_path.split("/")[-1].split(".mp4")[0]
        video_name=data_path.split("/")[-1]
        if dataset=="AUTSL":
            face_data_path = f"{train_face_cod_root}/{file_name}.csv"
            cod_data_path = f"{train_cod_root}/{file_name}.csv"
        else:
            face_data_path = f"{train_face_cod_root}/{gloss_dir}/{file_name}.csv"
            cod_data_path = f"{train_cod_root}/{gloss_dir}/{file_name}.csv"
        face_data=np.loadtxt(f"{face_data_path}",delimiter=",",dtype=np.float32)
        cod_data=np.loadtxt(f"{cod_data_path}",delimiter=",",dtype=np.float32)
        try:
            if len(cod_data.shape)==1:
                cod_data=cod_data.reshape(1,-1)
                face_data=face_data.reshape(1,-1)
            cod_data, face_cod_data, hand_cod_data, body_cod_data = coordinate_preprocess_3d(cod_data,
                                                                                             face_data,
                                                                                             is_face_connect=False,is_delete_nan=False,
                                                                                             is_limit_area=True)
        except:
            print(f"Error processing {data_path}. Skipping...")
            continue
        gloss_dir = os.path.basename(os.path.dirname(data_path))
        if not gloss_dir in gloss_data_dict.keys():
            gloss_data_dict[gloss_dir]=[]
        gloss_data_dict[gloss_dir].append(cod_data)
        c+=1
    print(f"num_glosses:{len(gloss_data_dict)}")
    for gloss in gloss_data_dict.keys():
        gloss_data=gloss_data_dict[gloss]
        if len(gloss_data)<2:
            print(f"gloss:{gloss} has only {len(gloss_data)} samples. Skipping...")
            continue
        C,T,J=gloss_data[0].shape
        gloss_data=[data.transpose(1,0,2).reshape(-1,J*C) for data in gloss_data]
        if len(gloss_data)==2:
            centroids_dict[gloss]={0:{"centroids":gloss_data[0],"data":gloss_data[0]},1:{"centroids":gloss_data[1],"data":gloss_data[1]}}
            continue
        elif len(gloss_data)==1:
            centroids_dict[gloss]={0:{"centroids":gloss_data[0],"data":gloss_data[0]}}
            continue
        else:
            gloss_data=to_time_series_dataset(gloss_data)
            sigma=estimate_sigma(gloss_data,n_samples=min(1000,len(gloss_data)))
            #K=build_gram(gloss_data,sigma)
            try:
                sigma_star,k_star=finalize_sigma_k(X,sigma)
            except Exception as e:
                print(f"Error in finalize_sigma_k for gloss:{gloss}. Using default sigma and k. Error: {e}")
                sigma_star=sigma
                #クラスタリングなし
                centroids_dict[gloss]={0:{"centroids":gloss_data[0],"data":gloss_data}}
                continue
            #k= eigengap_suggestion(K)
            K=build_gram(gloss_data,sigma_star)
            print(f"gloss:{gloss}, sigma:{sigma_star}, k:{k_star}")
            labels, T, centroids_embed, reps = spectral_with_centroids(K,k_star,random_state=42)
            #各クラスタのデータ数を取得
            cluster_count={i:np.sum(labels==i) for i in range(k_star)}
            #cluster_countの上位3個のクラスタを取得(クラスタ数が3未満の場合は全てのクラスタを取得)
            top_clusters=list(cluster_count.keys())
            embed_dict={i:{"centroids":gloss_data[reps[i]].reshape(-1,C*J),"data":gloss_data[labels==i]} for i in top_clusters}
            centroids_dict[gloss]=embed_dict

    with open(f"{save_path}/centroids_dict.pkl","wb") as f:
        pickle.dump(centroids_dict,f)


def clustering_skeleton(centroids_path,save_path):
    with open(centroids_path,"rb") as f:
        centroids_dict=pickle.load(f)
    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)
    for gloss in centroids_dict.keys():
        print(f"Processing gloss:{gloss}...")
        gloss_centroids=centroids_dict[gloss]
        if os.path.exists(f"{save_path}/{gloss}"):
            shutil.rmtree(f"{save_path}/{gloss}")
        os.makedirs(f"{save_path}/{gloss}",exist_ok=True)
        for cluster_id in gloss_centroids.keys():
            print(f"Processing cluster_id:{cluster_id}...")
            cluster_data=gloss_centroids[cluster_id]["data"]
            centroids_data=gloss_centroids[cluster_id]["centroids"]
            save_dir=f"{save_path}/{gloss}/{cluster_id}"
            os.makedirs(save_dir,exist_ok=True)
            try:
                B,T,JC=cluster_data.shape
            except:
                print(f"Error processing cluster_data for gloss:{gloss}, cluster_id:{cluster_id}. Skipping...")
                continue
            C=3
            J=JC//C
            cluster_data=cluster_data.reshape(B,T,C,J)
            cluster_data=np.where(np.isnan(cluster_data),0,cluster_data)
            #centroidsをプロット
            centroids_data=centroids_data.reshape(-1,C,J)
            centroids_data=np.where(np.isnan(centroids_data),0,centroids_data)
            video_writer=cv2.VideoWriter(f"{save_dir}/{gloss}_{cluster_id}_centroids.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                         30, (512, 512))
            for t in range(centroids_data.shape[0]):
                frame=np.ones((512,512,3),dtype=np.uint8)*255
                for j in range(J):
                    x=int(centroids_data[t,0,j]*512)
                    y=int(centroids_data[t,1,j]*512)
                    cv2.circle(frame,(x,y),3,(0,0,255),-1)
                video_writer.write(frame)
            video_writer.release()
            #opencvで座標をプロットし，動画にする
            for b in range(B):
                video_writer = cv2.VideoWriter(f"{save_dir}/{gloss}_{cluster_id}_{b}.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                               30, (512, 512))

                for t in range(T):
                    frame=np.ones((512,512,3),dtype=np.uint8)*255
                    for j in range(J):
                        x=int(cluster_data[b,t,0,j]*512)
                        y=int(cluster_data[b,t,1,j]*512)
                        cv2.circle(frame,(x,y),3,(0,255,0),-1)
                    video_writer.write(frame)
                video_writer.release()


if __name__=="__main__":
    dataset="phoenixT"
    save_path=f"{WORDS_DATADIR_T_SKELETON}"
    sign_video_feature_extractoion(dataset,save_path)
    clustering_skeleton(f"{save_path}/centroids_dict.pkl",f"{save_path}/clustered_skeletons")