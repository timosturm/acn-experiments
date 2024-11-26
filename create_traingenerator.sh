#!/bin/bash
for i in 5 10 20 50 80 100 150 200
do
    session="create_${i}kW"
    tmux new-session -ds "$session" "python create_traingenerator.py --transformer-cap $i"
    tmux pipe-pane -ot "$session" "cat > $PWD/${session}.ansi"
done