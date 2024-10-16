seed=$1

# One-Step Reach-Avoid
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_1_1_5 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalReachAvoid_1_1_1_5_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_1_1_5 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalReachAvoid_1_1_1_5_pretrained_RGCN_seed_${seed}.txt &

# Multi-Step Reach-Avoid
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_5_1_5 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/dfa_train_CompositionalReachAvoid_1_5_1_5_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 20000000 --discount 0.94 --ltl-sampler CompositionalReachAvoid_1_5_1_5 --epochs 4 --lr 0.0003 --seed ${seed} --dfa --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/dfa_train_CompositionalReachAvoid_1_5_1_5_pretrained_RGCN_seed_${seed}.txt &

wait
