"""
Script to run VIBE on a folder of cropped and centred images - basically same functionality as
demo.py, except without the multi-person tracking and bounding box cropping.
"""

import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import time
import torch
import pickle
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from smplx import SMPL

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import InferenceFromCrops
from lib.data_utils.kp_utils import convert_kps

from lib.utils.demo_utils import (
    smplify_runner,
    prepare_rendering_results,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        device=device,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file, map_location=device)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    total_time = time.time()
    # ========= Run VIBE on crops ========= #
    print(f'Running VIBE on crops...')
    vibe_time = time.time()
    image_folder = args.input_folder

    dataset = InferenceFromCrops(image_folder=image_folder)
    orig_height = orig_width = 512

    dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=0)

    with torch.no_grad():

        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

        for batch_num, batch in enumerate(dataloader):
            print("BATCH:", batch_num)
            batch = batch.unsqueeze(0)
            batch = batch.to(device)

            batch_size, seqlen = batch.shape[:2]
            output = model(batch)[-1]

            pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))


        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)

        del batch

    # ========= [Optional] run Temporal SMPLify to refine the results ========= #
    if args.run_smplify and args.tracking_method == 'pose':
        norm_joints2d = np.concatenate(norm_joints2d, axis=0)
        norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
        norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

        # Run Temporal SMPLify
        update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
        new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
            pred_rotmat=pred_pose,
            pred_betas=pred_betas,
            pred_cam=pred_cam,
            j2d=norm_joints2d,
            device=device,
            batch_size=norm_joints2d.shape[0],
            pose2aa=False,
        )

        # update the parameters after refinement
        print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
        pred_verts = pred_verts.cpu()
        pred_cam = pred_cam.cpu()
        pred_pose = pred_pose.cpu()
        pred_betas = pred_betas.cpu()
        pred_joints3d = pred_joints3d.cpu()
        pred_verts[update] = new_opt_vertices[update]
        pred_cam[update] = new_opt_cam[update]
        pred_pose[update] = new_opt_pose[update]
        pred_betas[update] = new_opt_betas[update]
        pred_joints3d[update] = new_opt_joints3d[update]

    elif args.run_smplify and args.tracking_method == 'bbox':
        print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
        print('[WARNING] Continuing without running Temporal SMPLify!..')

    # ========= Save results to a pickle file ========= #
    output_path = image_folder.replace('cropped_frames', 'vibe_results')
    os.makedirs(output_path, exist_ok=True)

    pred_cam = pred_cam.cpu().numpy()
    pred_verts = pred_verts.cpu().numpy()
    pred_pose = pred_pose.cpu().numpy()
    pred_betas = pred_betas.cpu().numpy()
    pred_joints3d = pred_joints3d.cpu().numpy()

    vibe_results = {
        'pred_cam': pred_cam,
        'verts': pred_verts,
        'pose': pred_pose,
        'betas': pred_betas,
        'joints3d': pred_joints3d,
    }

    del model
    end = time.time()
    fps = len(dataset) / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {len(dataset) / total_time:.2f}.')

    print(f'Saving vibe results to \"{os.path.join(output_path, "vibe_results.pkl")}\".')

    with open(os.path.join(output_path, "vibe_results.pkl"), 'wb') as f_save:
        pickle.dump(vibe_results, f_save)

    if not args.no_render:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = os.path.join(output_path, 'vibe_images')
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            frame_verts = vibe_results['verts'][frame_idx]
            frame_cam = vibe_results['pred_cam'][frame_idx]

            mesh_filename = None

            if args.save_obj:
                mesh_folder = os.path.join(output_path, 'vibe_meshes')
                os.makedirs(mesh_folder, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            rend_img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                mesh_filename=mesh_filename,
            )

            whole_img = rend_img

            if args.sideview:
                side_img_bg = np.zeros_like(img)
                side_rend_img90 = renderer.render(
                    side_img_bg,
                    frame_verts,
                    cam=frame_cam,
                    angle=90,
                    axis=[0,1,0],
                )
                side_rend_img270 = renderer.render(
                    side_img_bg,
                    frame_verts,
                    cam=frame_cam,
                    angle=270,
                    axis=[0, 1, 0],
                )
                if args.reposed_render:
                    smpl = SMPL('data/vibe_data',
                                batch_size=1)
                    zero_pose = torch.from_numpy(np.zeros((1, pred_pose.shape[-1]))).float()
                    zero_pose[:, 0] = np.pi
                    pred_frame_betas = torch.from_numpy(pred_betas[frame_idx][None,:]).float()
                    with torch.no_grad():
                        reposed_smpl_output = smpl(betas=pred_frame_betas,
                                                   body_pose=zero_pose[:, 3:],
                                                   global_orient=zero_pose[:, :3])
                        reposed_verts = reposed_smpl_output.vertices
                        reposed_verts = reposed_verts.cpu().detach().numpy()

                    reposed_cam = np.array([0.9, 0, 0])
                    reposed_rend_img = renderer.render(side_img_bg,
                                                       reposed_verts[0],
                                                       cam=reposed_cam)
                    reposed_rend_img90 = renderer.render(side_img_bg,
                                                         reposed_verts[0],
                                                         cam=reposed_cam,
                                                         angle=90,
                                                         axis=[0, 1, 0])

                    top_row = np.concatenate([img, reposed_rend_img, reposed_rend_img90],
                                             axis=1)
                    bot_row = np.concatenate([rend_img, side_rend_img90, side_rend_img270],
                                             axis=1)
                    whole_img = np.concatenate([top_row, bot_row], axis=0)

                else:
                    top_row = np.concatenate([img, side_img_bg, side_img_bg],
                                             axis=1)
                    bot_row = np.concatenate([rend_img, side_rend_img90, side_rend_img270],
                                             axis=1)
                    whole_img = np.concatenate([top_row, bot_row], axis=0)

            # cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), whole_img)
            cv2.imwrite(os.path.join(output_img_folder, os.path.basename(img_fname)),
                        whole_img)

        # ========= Save rendered video ========= #
        save_vid_path = os.path.join(output_path, 'vibe_video.mp4')
        print(f'Saving result video to {save_vid_path}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_vid_path)

    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=str,
                        help='input folder of cropped and centred images.')

    parser.add_argument('--vibe_batch_size', type=int, default=32,
                        help='batch size of VIBE')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--reposed_render', action='store_true',
                        help='render reposed meshes.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()

    main(args)
