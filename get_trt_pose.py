BASE_URL = "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master"

def download(url, dest):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to: {dest}")

MODEL_PATHS = {
    "resnet18": "resnet18_baseline_att_224x224_A_epoch_249.pth",
}

def main():
    ...
    pth_url = f"{BASE_URL}/models/{MODEL_PATHS[args.model]}"
    json_url = f"{BASE_URL}/tasks/human_pose/human_pose.json"
