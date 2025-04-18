# Example:
# bash to_jit.sh gr1 aug22-test

robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_wb"

proj_name="${robot_name}_wb"
exptid=${2}

# Run the training script
python save_jit_5dof.py --robot "${robot_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --checkpoint 6000 \
