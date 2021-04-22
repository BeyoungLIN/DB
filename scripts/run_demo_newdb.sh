
export CUDA_VISIBLE_DEVICES=0

python demo.py \
experiments/seg_detector/fakepages0310_resnet50_deform_thre.yaml \
--image_path /disks/sdb/projs/AncientBooks/data/Dingxiu_test_newdb/0A0D3D1A7FC14756978AF46C4AE2F907/ \
--visualize \
--sort_boxes \
--resume \
/disks/sdc/projs/AncientBooks/models/db/fakepage_res50_iter3.bin \
--box_thresh 0.5 \
--result_dir /disks/sdc/euphoria/Dingxiu_db_test/Dingxiu_2_demo_results_res50_2 \
