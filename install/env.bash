conda create -n heygenclone python=3.10.8
conda activate heygenclone

current_folder=$(dirname "$(realpath "$0")")

pip install -r "$current_folder/requirements/part1.txt"
pip install -r "$current_folder/requirements/part2.txt"