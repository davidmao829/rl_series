# Example:
# bash eval.sh gr1 aug22-test
# bash eval.sh h1 aug22-test
# bash eval.sh gr2t4 aug22-test

robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_explicit"

proj_name="${robot_name}_explicit"
exptid=${2}

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 64 \
                # --record_video \
                # --checkpoint 4900 \
                # --use_jit \
                # --teleop_mode