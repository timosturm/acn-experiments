a=50
d=A
e=1

ent_coef=0.0001
learning_rate=0.0003

session="search_t=${a}_r=${d}_e=${e}"
session="${session//"."/$"dot"}"

# tmux new-session -ds "$session" "python run_search.py --transformer-cap $a --reward-cfg $d"
# tmux pipe-pane -ot ${session} "cat > $PWD/${session}.ansi"

sbatch -J $session run.sh $a $d $ent_coef $learning_rate $e
