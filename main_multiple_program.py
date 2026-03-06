import subprocess
import os
import time
if __name__=="__main__":
    # subprocess.run(["python", "main.py", "--cfg", "configs/visualconv/visualconv_kinetics400.yaml"])
    command = ['sudo', 'systemctl', 'stop', 'systemd-oomd']
    print("OOM killerを無効化")
    subprocess.run(command, input=("gazouken\n").encode(), check=True)
    print("無効化完了")
    #print("---PhoenixT---")
    subprocess.run(["/home/caffe/anaconda3/envs/SLG/bin/python", "main_t2p.py"])
    time.sleep(10)
    subprocess.run(["/home/caffe/anaconda3/envs/SLG/bin/python", "main_t2p_diffusion.py"])
    #print("main_VAC.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python", "main_VAC.py"])
    #time.sleep(10)
    #print("main_CSLR_distil.py  を実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python", "main_CSLR_distil.py"])
    #time.sleep(10)




    #print("main_ISLR_s3d.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python","main_ISLR_s3d.py"])
    #print("main_ISLR.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python","main_ISLR.py"])

    #print("main_VAC_generative.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python","main_VAC_generative.py"])
    #print("main_pretrain_generative_short.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python","main_pretrain_generative_short.py"])
    #print("main_pretrain_generative_long.pyを実行")
    #subprocess.run(["/home/caffe/anaconda3/envs/MAE_csr/bin/python","main_pretrain_generative_long.py"])
    time.sleep(10)
    print("全学習が終了しました．PCをシャットダウンします...")
    os.system("shutdown -h now")

    # subprocess.run(["python", "main.py", "--cfg", "configs/visualconv/visualconv_kinetics400.yaml","--num_classes","0"])