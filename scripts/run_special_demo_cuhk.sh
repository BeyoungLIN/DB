
export CUDA_VISIBLE_DEVICES=1

python special_demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sdb/euphoria/CUHK_OCR/3.17竞赛数据/测试图片 \
--resume \
/disks/sdc/projs/AncientBooks/models/db/fakepage_res50_iter3.bin \
--box_thresh 0.5 \
--sort_boxes \