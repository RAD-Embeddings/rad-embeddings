seed=$1

# No pretraining
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalGeneralDFA --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_train_CompositionalGeneralDFA_GATv2Conv_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_train_CompositionalReachAvoid_1_2_1_3_GATv2Conv_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoidFix_1_2_1_2 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_train_CompositionalReachAvoidFix_1_2_1_2_GATv2Conv_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalParity_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_train_CompositionalParity_1_2_1_3_GATv2Conv_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_train_CompositionalEventually_1_4_1_3_GATv2Conv_seed_${seed}.txt &

# Pretraining
python train_agent.py --algo ppo --env Simple-LTL-Env-v0 --log-interval 1 --save-interval 20 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler CompositionalGeneralDFA --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --epochs 2 --seed ${seed} --dfa --gnn GATv2Conv &> outputs/dfa_pretrain_GATv2Conv_seed_${seed}.txt;
cp -r storage/GATv2Conv-dumb_ac_CompositionalGeneralDFA_Simple-LTL-Env-v0_seed:${seed}_epochs:2_bs:1024_fpp:512_dsc:0.9_lr:0.001_ent:0.01_clip:0.1_prog:full_dfa:True symbol-storage/

# General DFA
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalGeneralDFA --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalGeneralDFA_pretrained_GATv2Conv_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalGeneralDFA --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalGeneralDFA_pretrained_GATv2Conv_seed_${seed}.txt &

# Reach-Avoid
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalReachAvoid_1_2_1_3_pretrained_GATv2Conv_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalReachAvoid_1_2_1_3_pretrained_GATv2Conv_seed_${seed}.txt &

# Reach-Avoid-Fix
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoidFix_1_2_1_2 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalReachAvoidFix_1_2_1_2_pretrained_GATv2Conv_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoidFix_1_2_1_2 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalReachAvoidFix_1_2_1_2_pretrained_GATv2Conv_seed_${seed}.txt &

# Parity
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalParity_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalParity_1_2_1_3_pretrained_GATv2Conv_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalParity_1_2_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalParity_1_2_1_3_pretrained_GATv2Conv_seed_${seed}.txt &

# Partially-Ordered
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalEventually_1_4_1_3_pretrained_GATv2Conv_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_4_1_3 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --gnn GATv2Conv --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalEventually_1_4_1_3_pretrained_GATv2Conv_seed_${seed}.txt &
