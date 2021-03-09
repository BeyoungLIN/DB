
export CUDA_VISIBLE_DEVICES=1

python demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path /disks/sdb/euphoria/CUHK_OCR/Training_Set/tif/ \
--visualize \
--sort_boxes \
--resume models/fakepages_resnet18 \
--box_thresh 0.5 \
--result_dir ./CUHK_demo_results \