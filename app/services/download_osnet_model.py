
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = (
    "osnet_x1_0_msmt17_combineall_256x128_"
    "amsgrad_ep150_stp60_lr0.0015_b64_fb10_"
    "softmax_labelsmooth_flip_jitter.pth"
)

model_path = hf_hub_download(
    repo_id="kaiyangzhou/osnet",
    filename=MODEL_NAME,
    local_dir=MODEL_DIR
)

print("OSNet model downloaded successfully!")
print("Model location:", model_path)
