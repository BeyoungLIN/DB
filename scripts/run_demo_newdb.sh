
export CUDA_VISIBLE_DEVICES=3

python demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sdd/beyoung/Fakepages/data/book_pages_big_circle/imgs_vertical \
--visualize \
--sort_boxes \
--resume \
models/fakepage_res50_iter2.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/db_test_results/big_circle/res50_2 \
