#!python3
import argparse
import os
import time

import math
import torch
import cv2
import numpy as np
import itertools

from experiment import Structure, Experiment
from concern.config import Configurable, Config
import utils

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show images eagerly')
    parser.add_argument('--sort_boxes', action='store_true', dest='sort_boxes',
                        help='Sort boxes for further works')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    # Delete train settings, prevent of reading training dataset
    experiment_args.pop('train')
    experiment_args.pop('evaluation')
    experiment_args.pop('validation')
    experiment = Configurable.construct_class_from_config(experiment_args)

    demo_handler = Demo(experiment, experiment_args, cmd=args)

    if os.path.isdir(args['image_path']):
        img_cnt = len(os.listdir(args['image_path']))
        for idx, img in enumerate(os.listdir(args['image_path'])):
            t = time.time()
            demo_handler.inference(os.path.join(args['image_path'], img), args['visualize'])
            print("{}/{} elapsed time : {:.4f}s".format(idx + 1, img_cnt, time.time() - t))
    else:
        t = time.time()
        demo_handler.inference(args['image_path'], args['visualize'])
        print("elapsed time : {}s".format(time.time() - t))


class Demo:
    def __init__(self, experiment, args, cmd=None):
        if cmd is None:
            cmd = dict()
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        # model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.init_torch_tensor()
        self.model = self.init_model()
        self.resume(self.model, self.model_path)
        self.model.eval()

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                if self.args['sort_boxes']:
                    new_boxes = []
                    # new_scores = []
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        new_boxes.append(boxes[i, :, :])
                        # new_scores.append(score)
                    recs = [utils.trans_poly_to_rec(idx, box) for idx, box in enumerate(new_boxes)]
                    cluster_rec_ids = utils.cluster_recs(recs)
                    cluster_recs = []
                    for k in cluster_rec_ids.keys():
                        box_ids = cluster_rec_ids[k]
                        cluster_recs.append([recs[box_id] for box_id in box_ids])
                    classified_recs = sorted(cluster_recs, key=utils.list_sort, reverse=True)
                    '''
                    new_classified_recs = []
                    for box_list in classified_recs:
                        new_classified_recs.append(sorted(box_list, key=utils.box_sort, reverse=False))
                    '''
                    classified_recs = [sorted(l, key=utils.box_sort, reverse=False) for l in classified_recs]
                    output_idxs = utils.read_out(classified_recs, recs)
                    output_idxs = [i.idx for i in output_idxs]
                    with open(result_file_path, 'wt') as res:
                        for idx in output_idxs:
                            box = new_boxes[idx].reshape(-1).tolist()
                            box = list(map(str, box))
                            result = ",".join(box)
                            res.write(result + "\n")
                else:
                    with open(result_file_path, 'wt') as res:
                        for i in range(boxes.shape[0]):
                            score = scores[i]
                            if score < self.args['box_thresh']:
                                continue
                            box = boxes[i, :, :].reshape(-1).tolist()
                            result = ",".join([str(int(x)) for x in box])
                            res.write(result + ',' + str(score) + "\n")

    def inference(self, image_path, visualize=False):
        # all_metrics = {}
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output, self.args['box_thresh'])
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0] + '.jpg'),
                            vis_image)


if __name__ == '__main__':
    main()
