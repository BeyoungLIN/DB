
export CUDA_VISIBLE_DEVICES=2

python demo.py \
experiments/seg_detector/fakepages0310_resnet50_deform_thre.yaml \
--image_path /disks/sdc/euphoria/single_pic \
--visualize \
--sort_boxes \
--resume \
/disks/sdc/projs/AncientBooks/models/db/fakepage_res50_iter3.5.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/single_pic_res/res50_3.5 \
