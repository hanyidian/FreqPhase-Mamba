# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model,ckpt_path)
    model = model.to(device) #在build_sam2中已经把model放在device上了
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model




def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # 处理checkpoint格式：如果是包装格式 {'model': state_dict}，则提取state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"加载包装格式checkpoint，包含 {len(state_dict)} 个参数")
        else:
            state_dict = checkpoint
            print(f"加载直接格式checkpoint，包含 {len(state_dict)} 个参数")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"警告: 缺少以下键: {missing_keys[:5]}... (共{len(missing_keys)}个)")
            # 不抛出错误，继续运行
            # logging.warning(f"Missing keys: {missing_keys}")

        if unexpected_keys:
            print(f"警告: 发现意外键: {unexpected_keys[:5]}... (共{len(unexpected_keys)}个)")
            # 不抛出错误，继续运行
            # logging.warning(f"Unexpected keys: {unexpected_keys}")

        print("Checkpoint加载完成（允许参数不完全匹配）")
        # if 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # else:
        #     logging.warning('Optimizer state not found in checkpoint')
        # if 'scheduler' in checkpoint:
        #     scheduler.load_state_dict(checkpoint['scheduler'])
        # else:
        #     logging.warning('Scheduler state not found in checkpoint')
        # start_epoch = checkpoint.get('epoch',0)
        # loss = checkpoint.get('loss',None)

        logging.info(f'Loaded checkpoint sucessfully from {ckpt_path}.')
    else:
        logging.info('No checkpoint path provided. Starting train from scratch')
        return 0,None
