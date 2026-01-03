python ../../train_rlpd_sim.py \
    --learner \
    --seed 0 \
    --batch_size 256 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet18-pretrained \
    --checkpoint_period 5000 \
    --checkpoint_path actor_ckpt/  



