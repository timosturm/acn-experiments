#!/bin/bash
for a in 10 30 150 300
do
    for b in 0.5 1 10 20
    do
        for c in 0.25 1 2 3
        do
            for d in A B C D
            do
                session="search_t=${a}_f=${b}_d=${c}_r=${d}"
                session="${session//"."/$"dot"}"

                tmux new-session -ds "$session" "python run_search.py --transformer-cap $a --frequency-multiplicator $b --duration-multiplicator $c --reward-cfg $d"
                tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"
            done
        done
    done
done