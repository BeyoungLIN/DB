
export CUDA_VISIBLE_DEVICES=0

python demo.py \
experiments/seg_detector/fakepages0310_resnet50_deform_thre.yaml \
--image_path /disks/sdb/euphoria/CUHK_OCR/3.22竞赛数据/imgs/ \
--visualize \
--sort_boxes \
--resume \
/disks/sdc/projs/AncientBooks/models/db/fakepage_res50_iter3.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/CUHK_db_test/CUHK_0322_demo_results_res50_3.5/ \
