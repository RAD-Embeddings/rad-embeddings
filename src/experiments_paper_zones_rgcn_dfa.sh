seed=$1

# No pretraining
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalGeneralDFA --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} &> outputs/zones_dfa_train_CompositionalGeneralDFA_RGCN_seed_${seed}.txt &
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalParity_1_2_1_3 --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} &> outputs/zones_dfa_train_CompositionalParity_1_2_1_3_RGCN_seed_${seed}.txt &

# Pretraining
python train_agent.py --algo ppo --env Simple-LTL-Zones-Env-v0 --log-interval 1 --save-interval 20 --frames-per-proc 512 --batch-size 1024 --frames 10000000 --dumb-ac --discount 0.9 --ltl-sampler CompositionalGeneralDFA --lr 0.001 --clip-eps 0.1 --gae-lambda 0.5 --epochs 2 --seed ${seed} --dfa &> outputs/zones_dfa_pretrain_RGCN_seed_${seed}.txt;
cp -r storage/RGCN_8x32_ROOT_SHARED-dumb_ac_CompositionalGeneralDFA_Simple-LTL-Zones-Env-v0_seed:${seed}_epochs:2_bs:1024_fpp:512_dsc:0.9_lr:0.001_ent:0.01_clip:0.1_prog:full_dfa:True symbol-storage/

# General DFA
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalGeneralDFA --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/zones_dfa_train_CompositionalGeneralDFA_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalGeneralDFA --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/zones_dfa_train_CompositionalGeneralDFA_pretrained_RGCN_seed_${seed}.txt &

# Parity
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalParity_1_2_1_3 --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA --freeze-ltl &> outputs/zones_dfa_train_CompositionalParity_1_2_1_3_pretrained_RGCN_frozen_seed_${seed}.txt &
python train_agent.py --algo ppo --env Zones-5-v0 --ltl-sampler CompositionalParity_1_2_1_3 --frames-per-proc 4096 --batch-size 2048 --lr 0.0003 --discount 0.998 --entropy-coef 0.003 --log-interval 1 --save-interval 2 --epochs 10 --frames 40000000 --dfa --seed ${seed} --pretrained-gnn --pretrained-gnn-sampler CompositionalGeneralDFA &> outputs/zones_dfa_train_CompositionalParity_1_2_1_3_pretrained_RGCN_seed_${seed}.txt &
