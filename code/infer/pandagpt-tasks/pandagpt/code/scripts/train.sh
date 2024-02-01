#!/bin/bash
#此参数用于指定运行作业的名称
#DSUB -n 8-yx-KWS

#此处需要把“用户名”修改为用户的用户名，例如用户名为gpuuser001 则此行写为“#DSUB -A root.bingxing2.gpuuser001”
#DSUB -A root.bingxing2.gpuuser590
 
#默认参数，一般不需要修改  
#DSUB -q root.default
#DSUB -l wuhanG5500
 
#跨节点任务不同类型程序job_type会有差异，请参考下文对应跨节点任务模板编写
#DSUB --job_type cosched
 
# 此参数用于指定资源。如申请 6核CPU，1卡GPU，48GB内存。
# cpu=6;gpu=1;mem=45000
# cpu=48;gpu=8;mem=360000
#DSUB -R 'cpu=48;gpu=8;mem=360000'

# 此参数用于指定运行作业的机器数量。单节点作业则为 1 。
#DSUB -N 1
 
# 此参数用于指定日志的输出，%J表示JOB_ID。
#DSUB -e ./paralog/%J.out
#DSUB -o ./paralog/%J.out

#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境(pytorch环境需要自己部署)
module load anaconda/2021.11
module load cuda/11.6
module load cudnn/8.8.1_cuda11.x
module load gcc/8.3.0-gcc-4.8.5-cpp
source activate pandagpt

# 创建状态文件，用于控制采集的进程
STATE_FILE="./paralog/state_${BATCH_JOB_ID}"
/usr/bin/touch ${STATE_FILE}

# 后台循环采集，每间隔 1s 采集一次GPU数据。
# 采集的数据将输出到本地 gpu_作业ID.log 文件中
function gpus_collection(){
while [[ `cat "${STATE_FILE}" | grep "over" | wc -l` == "0" ]]; do
/usr/bin/sleep 2
/usr/bin/nvidia-smi >> "./paralog/gpu_${BATCH_JOB_ID}.log"
done
}
gpus_collection &

# run
# python train.py --batch_size 64 --max_epochs 500 --num_workers 4 \

#                 --lora --lora_modality_names vision text \
#                 --lora_layer_idxs 1 2 3 4 5 6 7 8 \
#                 --self_contrast --datasets "cifar100" \
#                 --device cuda:0 --headless
# python classfication_match_demo_cifar100.py

#python train_cifar100.backup.py --batch_size 64 --max_epochs 550 --num_workers 4 \
#                --lora --lora_modality_names vision text \
#                --lora_layer_idxs 1 2 3 4 5 6 7 8 \
#                --lr 1e-6 \
#                --device cuda:0 --headless


#deepspeed --include localhost:0,1,2,3 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
#    --model openllama_peft \
#    --stage 1\
#    --data_path  /home/bingxing2/gpuuser590/yx/dataset/ImageCaption/train.json\
#    --image_root_path /home/bingxing2/gpuuser590/yx/dataset/ImageCaption/train2014/\
#    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
#    --vicuna_ckpt_path ../pretrained_ckpt/llama-7b/\
#    --max_tgt_len 400\
#    --save_path  ../ckpt/pandagpt_7b_v0_peft/\
#    --log_path ../ckpt/pandagpt_7b_v0_peft/log_rest/

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama_peft \
    --stage 1\
    --data_path  /home/bingxing2/gpuuser590/yx/dataset/KeyWordSpotting/train_select.json\
    --image_root_path /home/bingxing2/gpuuser590/yx/dataset/KeyWordSpotting\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/\
    --whisper_ckpt_path ../pretrained_ckpt/whisper_ckpt/medium-en/\
    --vicuna_ckpt_path ../pretrained_ckpt/llama-7b/\
    --max_tgt_len 400\
    --save_path  ../ckpt/pandagpt_7b_v2_peft/\
    --log_path ../ckpt/pandagpt_7b_v2_peft/log_rest/



# 关闭GPU采集进程
echo "over" >> "${STATE_FILE}"
