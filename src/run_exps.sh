# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 1 &> exps/dfa_comp_seed_1.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 2 &> exps/dfa_comp_seed_2.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 3 &> exps/dfa_comp_seed_3.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 4 &> exps/dfa_comp_seed_4.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler CompositionalEventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 5 &> exps/dfa_comp_seed_5.txt;

# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --seed 1 &> exps/ast_seed_1.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --seed 2 &> exps/ast_seed_2.txt;
# python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --seed 3 &> exps/ast_seed_3.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --seed 4 &> exps/ast_seed_4.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --seed 5 &> exps/ast_seed_5.txt;

python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 1 &> exps/dfa_mono_seed_1.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 2 &> exps/dfa_mono_seed_2.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 3 &> exps/dfa_mono_seed_3.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 4 &> exps/dfa_mono_seed_4.txt;
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 2000000 --discount 0.94 --ltl-sampler Eventually_1_5_1_4 --epochs 4 --lr 0.0003 --dfa --seed 5 &> exps/dfa_mono_seed_5.txt;
