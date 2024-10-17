##Setup
```
conda create --name rad-embeddings python=3.9.19
conda activate rad-embeddings
pip install pip==24.0
pip install -r requirements.txt
pip install -e src/envs/safety/safety-gym/
conda install -c conda-forge spot
python train_agent.py --algo ppo --env Letter-7x7-v3 --log-interval 1 --save-interval 20 --frames 10000000 --discount 0.94 --ltl-sampler CompositionalReach_1_2_1_2 --epochs 4 --lr 0.0003 --seed 1 --dfa --gnn GATv2Conv
```