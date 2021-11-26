
export CUDA_VISIBLE_DEVICES=2

python special_demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sde/euphoria/datasets/DingXiu/0A0E0571713B44C2B18978850E3180A9/000002.png \
--visualize \
--result_dir /disks/sde/euphoria/datasets/DingXiu_test/0A0E0571713B44C2B18978850E3180A9/ \
--resume \
models/fakepage_res50_iter2.bin \
--box_thresh 0.5 \
--sort_boxes \
