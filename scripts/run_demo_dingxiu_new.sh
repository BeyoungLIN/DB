
export CUDA_VISIBLE_DEVICES=0

python demo.py \
experiments/seg_detector/fakepages_resnet50_deform_thre.yaml \
--image_path /disks/sdb/projs/AncientBooks/data/Dingxiu_test_newdb/0A0CBAE0046F4AB7BCBFE12789547A78 \
--visualize \
--sort_boxes \
--resume \
/home/euphoria/pkg/seg_detector/models/fakepage_res50_iter2.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/dingxiu_new_db_test/dingxiu_new_demo_results_res50_2/ \

