
export CUDA_VISIBLE_DEVICES=0

python demo.py \
experiments/seg_detector/fakepages_resnet18_deform_thre.yaml \
--image_path /disks/sdb/euphoria/CUHK_OCR/3.22竞赛数据/imgs/image_032_meitu.jpg \
--visualize \
--sort_boxes \
--resume /disks/sdb/euphoria/pkg/seg_detector/models/fakepages_resnet18 \
--box_thresh 0.5 \
--result_dir /disks/sdb/euphoria/pkg/seg_detector/CUHK_0322_demo_results_res18_old_contrast \