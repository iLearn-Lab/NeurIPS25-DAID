#!/bin/bash
#SBATCH --job-name=remove_CADDM        # 作业名称
#SBATCH --output=./log/output_test_remove_50.log          # 标准输出日志文件
#SBATCH --error=./log/error_test_remove_50.log            # 错误日志文件
#SBATCH --nodes=1                    # 1 个计算节点
#SBATCH --ntasks=1                   # 1 个任务
#SBATCH --partition=gpu-long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=END,FAIL   # 在作业结束或失败时发送通知
#SBATCH --mail-user=HarryC@nus.edu.sg   # 指定接收通知的邮箱地址



# 加载环境（根据你的 Conda 位置调整）
module load cuda/12.1  # 如果你的集群使用 module 方式管理 CUDA
export WANDB_API_KEY=WANDB_API_KEY
# 激活 Conda
source ~/.bashrc       # 确保可以使用 conda 命令
conda activate df1  # 你的 Conda 环境名
#pip install torchcam
#pip install dlib
# 进入代码目录
cd /home/h/hchengxa/CADDM_test

# 运行 Python 代码
python test.py --cfg ./configs/caddm_test.cfg

# 结束后自动退出环境（可选）
conda deactivate