
export CUDA_VISIBLE_DEVICES=0

python train.py \
experiments/seg_detector/fakepages0310_resnet50_deform_thre.yaml \
--resume models/fakepage_res50_iter2.bin \
--num_gpus 1 \
--validate \