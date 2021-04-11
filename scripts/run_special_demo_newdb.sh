
export CUDA_VISIBLE_DEVICES=0

python special_demo.py \
experiments/seg_detector/fakepages0310_resnet50_deform_thre.yaml \
--image_path ../data/DingXiu \
--resume \
/disks/sdc/projs/AncientBooks/models/db/fakepage_res50_iter3.bin \
--box_thresh 0.5 \
--sort_boxes \
