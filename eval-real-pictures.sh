set -eux
find real-pictures/ -type f | xargs python3 eval_v2.py
