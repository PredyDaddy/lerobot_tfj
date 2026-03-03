# HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python - <<'PY'
import subprocess
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id="admin123/grasp_block_in_bin"
ep=11  # episode_index



cam="observation.images.top"  # 或 "observation.images.wrist"




meta=LeRobotDatasetMetadata(repo_id)
row=meta.episodes[ep]
mp4=str(meta.root / meta.get_video_file_path(ep, cam))
start=float(row[f"videos/{cam}/from_timestamp"])
end=float(row[f"videos/{cam}/to_timestamp"])
subprocess.run(["ffplay","-autoexit","-loglevel","error","-ss",f"{start:.3f}","-t",f"{end-start:.3f}",mp4])
# PY
