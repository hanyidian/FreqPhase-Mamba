CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "/root/autodl-tmp/Checkpoints/black/checkpoint_epoch_5.pth"
--test_image_path "/root/autodl-tmp/Datasets/black/test/image"
--test_gt_path "/root/autodl-tmp/Datasets/black/test/mask"
--save_path "/root/autodl-tmp/Datasets/black/test/predicted"

--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt"
--model_path "E:\Models\checkpoints\SAM2-UNet-Kvair(original)\best_model.pth"
--test_image_path "F:\datasets\polyp\kvasir\Kvasir-SEG\Kvasir-SEG\data\test\images"
--test_mask_path "F:\datasets\polyp\kvasir\Kvasir-SEG\Kvasir-SEG\data\test\labels"

--hiera_path /root/autodl-tmp/checkpoints/sam2.1_hiera_large.pt
    --model_path /root/autodl-tmp/checkpoints/1/best_model.pth
    --test_image_path your_dataset_root/test/images
    --test_mask_path your_dataset_root/test/labels
    --batch_size 8
    --img_size 352
    --max_test_samples_to_save 50


--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt" --model_path "E:\Models\checkpoints\polpy\clinicDB\best_model.pth" --test_image_path "F:\datasets\polyp\CVC-ClinicDB\PNG\data\test\images" --test_mask_path "F:\datasets\polyp\CVC-ClinicDB\PNG\data\test\masks"