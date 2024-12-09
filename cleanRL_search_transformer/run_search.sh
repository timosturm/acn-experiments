#!/bin/bash

# ent_coef=0.0001
# learning_rate=0.001
# for transformer_cap in 10 30 150 300
# do
#     for reward_cfg in A B C D
#     do
#         session="search_t=${transformer_cap}_r=${reward_cfg}"
#         session="${session//"."/$"dot"}"

#         tmux new-session -ds "$session" "python run_search.py --transformer-cap $transformer_cap --reward-cfg $reward_cfg --ent-coef $ent_coef --learning-rate $learning_rate"
#         tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"

#         # sbatch -J $session run.sh $transformer_cap $reward_cfg $ent_coef $learning_rate
#     done
# done

for ent_coef in 0.001 0.00001
do
    for learning_rate in 0.0001 0.00001
    do
        for transformer_cap in 80 100 200
        do
            reward_cfg=D

            session="search_t=${transformer_cap}_r=${reward_cfg}_ent_coef=${ent_coef}_learning_rate=${learning_rate}"
            session="${session//"."/$"dot"}"
            sbatch -J $session run.sh $transformer_cap $reward_cfg $ent_coef $learning_rate
        done

        for transformer_cap in 5 10 20 80
        do
            reward_cfg=C

            session="search_t=${transformer_cap}_r=${reward_cfg}_ent_coef=${ent_coef}_learning_rate=${learning_rate}"
            session="${session//"."/$"dot"}"
            sbatch -J $session run.sh $transformer_cap $reward_cfg $ent_coef $learning_rate
        done

        for transformer_cap in 5 20
        do
            reward_cfg=B

            session="search_t=${transformer_cap}_r=${reward_cfg}_ent_coef=${ent_coef}_learning_rate=${learning_rate}"
            session="${session//"."/$"dot"}"
            sbatch -J $session run.sh $transformer_cap $reward_cfg $ent_coef $learning_rate
        done
    done
done