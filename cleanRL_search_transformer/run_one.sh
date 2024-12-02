a=50
d=A

session="search_t=${a}_r=${d}"
session="${session//"."/$"dot"}"

tmux new-session -ds "$session" "python run_search.py --transformer-cap $a --reward-cfg $d"
tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"