seed=$1

# No pretraining
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Until_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa &> outputs/ltl_train_CompositionalReachAvoid_1_2_1_3_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} &> outputs/ltl_train_CompositionalEventually_1_4_1_3_seed_${seed}.txt &

# Pretraining
python train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 20 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler Eventually_1_4_1_3_JOIN_Until_1_2_1_3 --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --epochs 2 --seed ${seed} &> outputs/ltl_pretrain_RGCN_seed_${seed}.txt;
cp -r storage/RGCN_8x32_ROOT_SHARED-dumb_ac_Eventually_1_4_1_3_JOIN_Until_1_2_1_3_Simple-LTL-Env-v0_seed:${seed}_epochs:2_bs:1024_fpp:512_dsc:0.9_lr:0.001_ent:0.01_clip:0.1_prog:full_dfa:False symbol-storage/

# Until
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Until_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler Eventually_1_4_1_3_JOIN_Until_1_2_1_3 --freeze-ltl &> outputs/ltl_train_Until_1_2_1_3_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Until_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler Eventually_1_4_1_3_JOIN_Until_1_2_1_3 &> outputs/ltl_train_Until_1_2_1_3_pretrained_RGCN_seed_${seed}.txt &

# Eventually
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler Eventually_1_4_1_3_JOIN_Until_1_2_1_3 --freeze-ltl &> outputs/ltl_train_Eventually_1_4_1_3_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler Eventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler Eventually_1_4_1_3_JOIN_Until_1_2_1_3 &> outputs/ltl_train_Eventually_1_4_1_3_pretrained_RGCN_seed_${seed}.txt &
