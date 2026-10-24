#!/bin/bash
# Start sim2sim in one tmux session: the sim (MuJoCo window) in window "sim", the keyboard policy in window
# "policy", plus function keys that work from the policy terminal:
#   F9  release / attach the elastic band     F5  reset the sim to the spawn point (band unchanged)
#   F6  start the waypoint follower           F7  stop it (the policy falls back to zero velocity)
#
#   bash rpl_sim2sim/run_sim2sim_tmux.sh                # keyboard velocity (w/s, q/e)
#   bash rpl_sim2sim/run_sim2sim_tmux.sh --waypoints    # velocity from waypoint_follower.py (F6)
#   bash rpl_sim2sim/run_sim2sim_tmux.sh --spawn 2.5 -2 1.893   # start somewhere else (default: floor before stairs)
#   bash rpl_sim2sim/run_sim2sim_tmux.sh --teacher stairs_up    # height-map teacher instead of the depth student
#   tmux kill-session -t rpl_sim2sim                    # stop everything
#
# Extra follower options go in WAYPOINT_ARGS, e.g. WAYPOINT_ARGS="--route full" or "--return forward".
# Console logs: logs/sim2sim/console_{sim,policy,waypoints}.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="${CONDA_ROOT:-$HOME/.holosoma_deps/miniconda3}/envs"
SIM_PY="$ENVS/${SIM_ENV:-rplmujoco}/bin/python"
POLICY_PY="$ENVS/${POLICY_ENV:-rplinference}/bin/python"
SESSION=rpl_sim2sim
LOGS="$ROOT/logs/sim2sim"

SPAWN=(0 -2 0.8)
VELOCITY=keyboard
TEACHER=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --waypoints) VELOCITY=udp; shift ;;
        --spawn) SPAWN=("$2" "$3" "$4"); shift 4 ;;
        --teacher) TEACHER="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION exists; stop it first: tmux kill-session -t $SESSION" >&2; exit 1; }
mkdir -p "$LOGS"

# PYTHONPATH from a ROS setup breaks the conda envs (e.g. "No module named 'lark'").
# --teacher: the sim also sends the height map, and the policy runs that terrain's exported teacher
# (rpl_sim2sim/tools/export_teacher_onnx.py) instead of the depth student.
SIM_ARGS="--no-joystick --spawn ${SPAWN[*]}"
POLICY_PRESET=""
if [[ -n "$TEACHER" ]]; then
    MODEL="rpl_sim2sim/models/rpl_teacher_${TEACHER}_model_74998.onnx"
    [[ -f "$ROOT/$MODEL" ]] || { echo "no $MODEL; run rpl_sim2sim/tools/export_teacher_onnx.py --all" >&2; exit 1; }
    SIM_ARGS="$SIM_ARGS --height-scan"
    POLICY_PRESET="inference:g1-29dof-rpl-teacher --task.model-path $MODEL"
fi
SIM_CMD="env -u PYTHONPATH bash rpl_sim2sim/run_sim.sh $SIM_ARGS 2>&1 | tee $LOGS/console_sim.log"
POLICY_ENV_VARS="env -u PYTHONPATH"
[[ "$VELOCITY" == udp ]] && POLICY_ENV_VARS="env -u PYTHONPATH RPL_VELOCITY_INPUT=udp"
# The policy attaches to the sim's DDS and depth shared memory, so wait until the sim loop runs.
POLICY_CMD="until grep -q 'FPS:' $LOGS/console_sim.log 2>/dev/null; do sleep 1; done; \
$POLICY_ENV_VARS bash rpl_sim2sim/run_policy.sh $POLICY_PRESET --task.velocity-input keyboard --task.state-input keyboard \
2>&1 | tee $LOGS/console_policy.log; echo '[policy exited] press Enter'; read"
FOLLOWER_CMD="env -u PYTHONPATH $POLICY_PY -m rpl_sim2sim.waypoint_follower ${WAYPOINT_ARGS:-} 2>&1 \
| tee $LOGS/console_waypoints.log; echo '[waypoint follower exited] press Enter'; read"

rm -f "$LOGS/console_sim.log"
tmux new-session -d -s "$SESSION" -n policy -c "$ROOT" "$POLICY_CMD"
tmux new-window -d -t "$SESSION" -n sim -c "$ROOT" "$SIM_CMD; echo '[sim exited] press Enter'; read"

# The sim handles SIGUSR1 (reset) and SIGUSR2 (band); patterns start with ^ so they match only the python process.
SIM_PAT="^$SIM_PY -m rpl_sim2sim.run_sim"
FOLLOWER_PAT="^$POLICY_PY -m rpl_sim2sim.waypoint_follower"
tmux bind-key -n F5 run-shell "pkill -USR1 -f '$SIM_PAT'; tmux display-message 'sim reset to spawn (band unchanged)'"
tmux bind-key -n F9 run-shell "pkill -USR2 -f '$SIM_PAT'; tmux display-message 'elastic band toggled'"
if [[ "$VELOCITY" == udp ]]; then
    tmux bind-key -n F6 split-window -v -d -l 14 -t "$SESSION:policy" -c "$ROOT" "$FOLLOWER_CMD"
    tmux bind-key -n F7 run-shell "pkill -INT -f '$FOLLOWER_PAT'; tmux display-message 'waypoint follower stopped'"
    STATUS=" F6/F7 waypoints on/off | F9 band | F5 reset "
else
    tmux bind-key -n F6 display-message "started without --waypoints: velocity comes from w/s/q/e"
    tmux bind-key -n F7 display-message "started without --waypoints"
    STATUS=" F9 band | F5 reset | w/s q/e = velocity "
fi
tmux set-option -t "$SESSION" status-right "$STATUS"
tmux set-option -t "$SESSION" status-right-length 80

echo "started tmux session $SESSION (velocity: $VELOCITY, spawn: ${SPAWN[*]}, policy: ${TEACHER:+teacher }${TEACHER:-depth student})"
if [[ -t 1 ]]; then
    exec tmux attach -t "$SESSION:policy"
else
    echo "attach with: tmux attach -t $SESSION:policy"
fi
