import os
from pathlib import Path

suite = "metaworld"
obs_type = "pixels"
env = "hammer"
csv_path = Path("exp_local/2022.09.29/")
result_path = Path(f"./plot/results/{suite}_{obs_type}/{env}")
result_path.mkdir(parents=True, exist_ok=True)
method = "rot_explore1e1_weight10"

folders = os.listdir(csv_path)
for folder in folders:
    if method not in folder:
        continue
    file_path = csv_path / folder / "eval.csv"
    seed = int(folder.split("_")[-1])
    result_file = result_path / f"{method}_seed{seed}.csv"
    os.system(f"cp {file_path} {result_file}")
