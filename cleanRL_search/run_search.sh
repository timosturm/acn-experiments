#!/bin/bash
ent_coef=0.0001
learning_rate=0.001

for transformer_cap in 10 30 150 300
do
    for frequency_multiplicator in 0.5 1 10 20
    do
        for duration_multiplicator in 0.25 1 2 3
        do
            for reward_cfg in A B C D
            do
                session="search_t=${a}_f=${b}_d=${c}_r=${d}"
                session="${session//"."/$"dot"}"

                tmux new-session -ds "$session" "python run_search.py --transformer-cap $transformer_cap --frequency-multiplicator $frequency_multiplicator --duration-multiplicator $duration_multiplicator --reward-cfg $reward_cfg --ent-coef $ent_coef --learning-rate $learning_rate"
                tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"

		        # sbatch -J $session run.sh $transformer_cap $frequency_multiplicator $duration_multiplicator $reward_cfg $ent_coef $learning_rate
            done
        done
    done
done
