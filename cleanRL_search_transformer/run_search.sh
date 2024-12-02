#!/bin/bash
for a in 5 10 20 50 80 100 150 200
do
    for d in A B C D
    do
        session="search_t=${a}_r=${d}"
        session="${session//"."/$"dot"}"

        tmux new-session -ds "$session" "python run_search.py --transformer-cap $a --reward-cfg $d"
        tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"
    done
done