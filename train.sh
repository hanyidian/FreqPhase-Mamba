CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt"
--train_image_path "E:\Datasets\DRIVE\train\images"
--train_mask_path "E:\Datasets\DRIVE\train\labels"
--save_path "E:\Models\checkpoints\SAM2-UNet_eyes"
--epoch 20 \
--lr 0.001 \
--batch_size 12


CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/tmp/checkpoints/SAM2/sam2_hiera_large.pt"
--train_image_path "/tmp/Datasets/Brain/BraTS/Train/images"
--train_mask_path "/tmp/Datasets/Brain/BraTS/Train/masks"
--save_path "/tmp/checkpoints/Brain/BraTS"
--epoch 20 \
--lr 0.001 \
--batch_size 12

--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt" --train_image_path "E:\Datasets\vessel\FIves\train\Original" --train_mask_path "E:\Datasets\vessel\FIves\train\Ground truth" --save_path "E:\Models\checkpoints\SAM2-UNet_eyes"
--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt" --train_image_path "E:\Datasets\vessel\DRIVE\train\images" --train_mask_path "E:\Datasets\vessel\DRIVE\train\labels" --save_path "E:\Models\checkpoints\SAM2-UNet_eyes"
--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt" --train_image_path "E:\Datasets\polyp\kvasir\Kvasir-SEG\Kvasir-SEG\images" --train_mask_path "E:\Datasets\polyp\kvasir\Kvasir-SEG\Kvasir-SEG\masks" --save_path "E:\Models\checkpoints\SAM2-UNet_Kvair"
--hiera_path "E:\Models\checkpoints\SAM2\sam2.1_hiera_large.pt" --train_image_path "F:\datasets\polyp\CVC-ClinicDB\PNG\data\train\images" --train_mask_path "F:\datasets\polyp\CVC-ClinicDB\PNG\data\train\masks" --save_path "E:\Models\checkpoints\polpy\clinicDB"

