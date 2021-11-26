
export CUDA_VISIBLE_DEVICES=2

python demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sdb/projs/AncientBooks/data/DingXiu/0A0CBAE0046F4AB7BCBFE12789547A78/ \
--visualize \
--sort_boxes \
--resume \
models/fakepage_res50_iter2.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/db_test_results/dingxiu \
