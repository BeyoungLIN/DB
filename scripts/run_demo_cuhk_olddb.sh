
export CUDA_VISIBLE_DEVICES=1

python demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sdd/beyoung/data/ER007/20_19584 \
--visualize \
--resume \
/home/euphoria/pkg/seg_detector/models/fakepage_res50_iter2.bin \
--box_thresh 0.5 \
--sort_boxes \
--result_dir /disks/sdb/euphoria/DB/datasets/ER007_demo \

