#!/bin/bash

#SBATCH --nodes=1   # Number of nodes to use
#SBATCH --ntasks-per-node=8   # Use 8 processor cores per node 
#SBATCH --time=2-8:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --mem=128G   # Maximum memory per node
#SBATCH --gres=gpu:a100:2   # Required GPU hardware
#SBATCH --job-name="llm-exp"   # Job name to display in squeue
#SBATCH --mail-user=znyu@iastate.edu   # Email address
#SBATCH --mail-type=BEGIN   # Send an email when the job starts
#SBATCH --mail-type=END   # Send an email when the job ends
#SBATCH --mail-type=FAIL   # Send an email if the job fails

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load python/3.10.10-zwlkg4l

source /work/LAS/hygao-lab/znyu/GRPO/grpo/bin/activate

# pip3 install flash-attn --no-build-isolation

# cd verl/
# pip3 install -e .[vllm]
# cd ../

export WANDB_API_KEY="cef0853f1b3259a908b6907a40981bd4fd274708"

echo $WANDB_API_KEY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/work/LAS/hygao-lab/znyu/GRPO/data/s1K/train.parquet" \
    data.val_files="/work/LAS/hygao-lab/znyu/GRPO/data/s1K/test.parquet" \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    reward_model.reward_manager="majority_vote" \
    custom_reward_function.path=/work/LAS/hygao-lab/znyu/GRPO/utils/reward_score/s1K.py \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/work/LAS/hygao-lab/znyu/GRPO/LLM-Model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_s1k_qwen3_8B_grpo_maj_4096_1_run' \
    trainer.experiment_name='Qwen3-8B-s1k-grpo-maj-1-run' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@