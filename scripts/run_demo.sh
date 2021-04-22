
export CUDA_VISIBLE_DEVICES=1

python demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path /disks/sdb/projs/AncientBooks/data/Dingxiu_test_newdb/0A0D3D1A7FC14756978AF46C4AE2F907/ \
--visualize \
--sort_boxes \
--resume /disks/sdb/euphoria/pkg/seg_detector/models/fakepage_res18_iter2.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/Dingxiu_db_test/Dingxiu_2_demo_results_res18 \
