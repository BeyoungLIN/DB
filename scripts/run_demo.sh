
export CUDA_VISIBLE_DEVICES=2

python demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path /disks/sdc/euphoria/single_pic/ \
--visualize \
--sort_boxes \
--resume /home/euphoria/pkg/seg_detector/models/fakepage_res18_iter2.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/single_pic_res/res18_2.0 \
