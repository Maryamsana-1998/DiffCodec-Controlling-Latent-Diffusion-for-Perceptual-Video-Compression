import os
import cv2
import numpy as np
from models.util import create_model, load_state_dict
from PIL import Image
from test_utils import *
import argparse
import json
import warnings
from src.test.video_codec import *
from src.train.util import *
from pathlib import Path
warnings.filterwarnings("ignore")

def post_proc(pred,pred_path):
    img = pred[0][0]
    # If image is in [0, 1] range, convert to [0, 255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    resized_img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(pred_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))



VIDEO_DETAILS = {
    "Beauty": {
        "prompt": "A beautiful blonde girl smiling with pink lipstick with black background",
        "path": "Beauty"
    },
    "Jockey": {
        "prompt": "A man riding a brown horse, galloping through a green race track. The man is wearing a yellow and red shirt and also a yellow hat",
        "path": "Jockey"
    },
    "Bosphorus": {
        "prompt": "A man and a woman sitting together on a boat sailing in water. They are both wearing ties. There is also a red flag at end of boat",
        "path": "Bosphorus"
    },
    "ShakeNDry": {
        "prompt": "A German shepherd shakes off water in the middle of a forest trail",
        "path": "ShakeNDry"
    },
    "YachtRide": {
        "prompt": "A sleek black-and-wood luxury yacht cruises through calm blue waters, carrying three passengers — two in conversation at the front and one steering under a shaded canopy.",
        "path": "YachtRide"
    },
    "HoneyBee": {
        "prompt": "Honeybees hover among blooming purple flowers",
        "path": "HoneyBee"
    },
    "ReadySteadyGo": {
        "prompt": "The moment of launch at Türkiye Jokey, as jockeys and their horses surge out of the starting gates on a lush green turf",
        "path": "ReadySteadyGo"
    }
}

# ====== MAIN PROCESS FUNCTION ======
def process_video(video,model,args):
    print(f'Processing: {video}')
    prompt = VIDEO_DETAILS[video]['prompt']
    base = os.path.join(args.original_root, video)

    original_dir = os.path.join(base, 'images')
    intra_dir = os.path.join(base, 'benchmark', f'decoded_q{args.intra_quality}')
    intra_paths = get_png_paths(intra_dir)
    selected_intra = select_intra_frames_by_gop(intra_paths, args.gop)
    total = 96
    intra_indices = list(range(0, total, args.gop))
    inter_indices = [i for i in range(total) if i not in intra_indices]
    local_list = args.local_list
    local_pngs ={}
    intra_frames = [ selected_intra[i // args.gop] for i in range(total)]

    for local_type in local_list:
        if local_type == 'r1':
            local_pngs['r1'] = intra_frames[:-1]
        if local_type =='r2':
            local_pngs['r2'] = intra_frames[1:]
        if local_type == 'flow':
            flow_dir = os.path.join(base, 'optical_flow', f'optical_flow_gop_{args.gop}_raft')
            local_pngs['flow'] = sorted(os.path.join(flow_dir, f) 
                                        for f in os.listdir(flow_dir)
                                        if f.lower().endswith('.flo'))
        if local_type == 'flow_b':
            flow_dir = os.path.join(base, 'optical_flow_bwd', f'optical_flow_gop_{args.gop}_raft')
            local_pngs['flow_b'] = sorted(os.path.join(flow_dir, f) 
                                        for f in os.listdir(flow_dir)
                                        if f.lower().endswith('.flo'))

        if local_type =='depth':
            depth_dir = os.path.join(base, 'depth') 
            local_pngs['depth']= get_png_paths(depth_dir)


    pred_dir = os.path.join(args.pred_root, video) 
    os.makedirs(pred_dir, exist_ok=True)

    # for i, orig_path in enumerate(original_paths):
    for i in range(20):
        pred_path = os.path.join(
            pred_dir,
            f'im{(i + 1):05d}_pred.png'
        )

        # skip if already exists
        if os.path.exists(pred_path):
            print('Already processed:', pred_path, '\n', '*******')
            continue

        # Intra-coded frames
        if i % args.gop == 0:
            print('Intra Coded:', local_pngs['r1'][i], '\n')
            img = cv2.imread(local_pngs['r1'][i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                pred_path,
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
        else:
            # build list of local images and paths: prev intra, next intra, flow (and depth)
            try:
                local_paths =[]
                flow_path = ''
                for local_type in sorted(local_list):
                    if local_type not in ['flow', 'flow_b']:
                        # Handle non-flow conditions
                        img_path = local_pngs[local_type][i]
                        local_paths.append(img_path)

                # Handle flow / flow_b separately
                flow, flow_b = None, None
                flow_path,flow_path_b = None,None
                if 'flow' in local_list:
                    flow_path = local_pngs['flow'][inter_indices.index(i)]
                    flow = load_flo_file(flow_path)
                    flow = adaptive_weighted_downsample(flow, target_h=128, target_w=128)

                if 'flow_b' in local_list:
                    flow_path_b = local_pngs['flow_b'][inter_indices.index(i)]
                    flow_b = load_flo_file(flow_path_b)
                    flow_b = adaptive_weighted_downsample(flow_b, target_h=128, target_w=128)

                # Concatenate if both flows exist
                if flow is not None and flow_b is not None:
                    flow_combined = np.concatenate([flow, flow_b])
                elif flow is not None:
                    flow_combined = normalize_for_warping(flow)
                else:
                    pass  

            except Exception as e:
                print('error and skipping', e)
                continue

            print('Inter Coded with:', local_paths,flow_path,flow_path_b,'\n')

            # load local images
            local_images = []
            for p in local_paths:
                # print(p)
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                local_images.append(img)

            # call your processing function (preserves logic)

            if args.emb_global:

                global_image=local_images[0]
            else:
                global_image= None
            pred = [None]
            if 'flow' in local_list:
                pred = process_wrap(model, local_images,flow=flow_combined, prompt=prompt,global_image=global_image,ddim_steps=40)
            else:
                pred = process(model, local_images, prompt)
            # write out the prediction
            post_proc(pred,pred_path)
        
    metrics = eval_video(original_dir, pred_dir,video)
    return metrics


def eval_video(original_folder, pred_folder,video):

    original_paths = get_png_paths(original_folder)
    original_eval_images = []
    pred_eval_images = []

    for i in range(0, len(original_paths)):
            
            original_path = os.path.join(original_folder, f"frame_{i:04d}.png")
            pred_path = os.path.join(pred_folder, f"im{i+1:05d}_pred.png")
            print(pred_path)

            if os.path.exists(original_path) and os.path.exists(pred_path):
                original = Image.open(original_path).convert("RGB")
                pred = Image.open(pred_path).convert("RGB")

                original_eval_images.append(original)
                pred_eval_images.append(pred)
            else:
                print(f"Warning: Missing image for {video} frame {i}")

    if original_eval_images and pred_eval_images:
        metrics = calculate_metrics_batch(original_eval_images, pred_eval_images)
    else:
        print(f"No images found or incomplete data for video {video}. Skipping metrics.")

    return metrics

# ====== EXECUTE ======
def main():

    parser = argparse.ArgumentParser(description="Evaluate Video Metrics and Generate Predictions")
    parser.add_argument("--original_root", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--local_list", nargs='+', default=["r1", "r2", "flow","flow_b"],
                        help="Select one or more videos: Beauty, Jockey, Bosphorus")
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--intra_quality", type=int, default=4, choices=[1, 4, 8])
    parser.add_argument("--resolution", type=str, default="1080p", choices=["512p", "1080p"])
    parser.add_argument("--emb_global", default=False,action='store_true')
    parser.add_argument("--grid", type=int, default=None,
                help="Optional grid size to append (e.g., 3 for grid_3). If not set, no grid folder is appended.")

    # parser.add_argument("--with_depth", type=bool, default=False)
    args = parser.parse_args()
    # ====== MODEL SETUP ======

    print("loading model")
    model = create_model(args.config).cpu()
    ckpt = load_state_dict(args.ckpt, location='cuda')

    # filter out any unmatched keys
    model_keys = set(model.state_dict())
    filtered = {k: v for k, v in ckpt.items() if k in model_keys}
    model.load_state_dict(filtered)
    model = model.cuda()

    all_metrics = {}
    for vid in VIDEO_DETAILS:
        metrics = process_video(vid,model,args)
        all_metrics[vid] = metrics

    for video, metrics in all_metrics.items():
        print(f"{video} Metrics: {metrics}")

    metrics_json_path = os.path.join(args.pred_root, f"all_videos_metrics_{args.gop}_q{args.intra_quality}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nAll metrics saved to {metrics_json_path}")

if __name__ == "__main__":
    main()
