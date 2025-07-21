import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import yaml
import zlib

from cmp.utils import flowlib
from cmp.models.cmp import CMP


def decompress_zlib_array(path, shape, dtype=np.int16, scale=10.0):
    with open(path, 'rb') as f:
        raw = zlib.decompress(f.read())
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr.astype(np.float32) / scale


def pre_process_image(img_path, repeat, img_transform):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((512, 512))
    tensor = torch.from_numpy(np.array(image).astype(np.float32).transpose((2, 0, 1))) / 255.0
    tensor = img_transform(tensor)
    image = tensor.unsqueeze(0).repeat(repeat, 1, 1, 1).cuda()
    return image


def decode_flow(sparse_flow, mask_flow, image, model):
    sparse = torch.from_numpy(sparse_flow).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    mask = torch.from_numpy(mask_flow).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    model.set_input(image, torch.cat([sparse, mask], dim=1), None)
    tensor_dict = model.eval(ret_loss=False)
    flow = tensor_dict['flow_tensors'][0].cpu().numpy().squeeze().transpose(1, 2, 0)
    return flow


def main(args):
    # Load config
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    data_mean = config_data['data']['data_mean']
    data_div = config_data['data']['data_div']
    img_transform = transforms.Normalize(data_mean, data_div)

    # Initialize model
    model = CMP(config_data['model'], dist_model=False)
    model.load_state(args.checkpoint, 1, False)
    model.switch_to('eval')

    # Prepare input/output
    sparse_files = sorted([f for f in os.listdir(args.flow_dir) if f.endswith("_flow.zlib")])
    intra_frames = sorted(os.listdir(args.intra_dir))
    os.makedirs(args.output, exist_ok=True)
    gop = args.gop
    repeat = 1

    for i, sparse_file in enumerate(sparse_files):
        basename = sparse_file.replace("_flow.zlib", "")
        mask_file = basename + "_mask.zlib"

        sparse_path = os.path.join(args.flow_dir, sparse_file)
        mask_path = os.path.join(args.mask_dir, mask_file)
        intra_index = (i + 1) // gop
        intra_path = os.path.join(args.intra_dir, intra_frames[intra_index])

        sparse = decompress_zlib_array(sparse_path, shape=(512, 512, 2))
        mask = decompress_zlib_array(mask_path, shape=(512, 512, 1))
        intra_img = pre_process_image(intra_path, repeat, img_transform)

        recon_flow = decode_flow(sparse, mask, intra_img, model)
        output_path = os.path.join(args.output, f"{basename}_recon.flo")
        flowlib.write_flow(recon_flow, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intra_dir", required=True, help="Directory containing intra frames")
    parser.add_argument("--flow_dir", required=True, help="Directory with compressed sparse flows")
    parser.add_argument("--output", required=True, help="Output directory for reconstructed flows")
    parser.add_argument("--config", required=True,
                        default='cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml', help="Path to CMP config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        default='cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/', help="Checkpoint directory for CMP")
    parser.add_argument("--gop", type=int, default=12, help="Group of pictures interval")
    args = parser.parse_args()
    main(args)
