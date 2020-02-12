import json
from tqdm import tqdm
from os.path import join as joinpath
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import colorsys
import random
import os
from collections import Counter
import math
import sys


# [line_color, style, linewidth, alpha]
BOX_TYPE_DICT = {
    'dotted': ['gray', 'dotted', 2, 0.5],
    'default': ['', 'solid', 2, 1],
    'strong': ['', 'solid', 4, 1]
}


class GQADataset:

    def __init__(self, data_root):

        self.val_sGraph = json.load(open(joinpath(data_root, 'val_sceneGraphs.json')))
        self.train_sGraph = json.load(open(joinpath(data_root, 'train_sceneGraphs.json')))
        self.all_sGraph = dict([(k, v) for k, v in list(self.val_sGraph.items()) + list(self.train_sGraph.items())])
        self.img_dir = joinpath(data_root, 'images')

        # obj_dict, attr_set, relation_set, img_set = process_scene_graph([joinpath(fp, 'val_sceneGraphs.json'),
        #                                                                  joinpath(fp, 'train_sceneGraphs.json')])

        self.obj2name, self.name2obj = {}, {}
        self.preprocess_gqa()

    def preprocess_gqa(self):
        for k, v in self.all_sGraph.items():
            img_obj_dict = v['objects']

            for obj_id, obj_info in img_obj_dict.items():
                name = obj_info['name']
                self.obj2name[obj_id] = name
                if name in self.name2obj:
                    self.name2obj[name].append(obj_id)
                else:
                    self.name2obj[name] = [obj_id]



    def get_sGraph(self, img_id):
        return self.all_sGraph[img_id]

    def get_flatRel(self, img_id, filter_set=None, filter_lr=False):
        """

        :param img_id:
        :param filter_set:
            set of obj names. If not None, only return relations that involve objs in the set
        :return:
        """

        sGraph = self.get_sGraph(img_id)
        img_obj_dict = sGraph['objects']

        rel_dict = dict()

        name_set = {v['name'] for k,v in img_obj_dict.items()}
        if filter_set is not None:
            if not all([obj in name_set for obj in filter_set]):
                return rel_dict

        for sub_id, sub_info in img_obj_dict.items():
            rel_ls = sub_info['relations']
            for e in rel_ls:
                rel_name, rel_obj_id = e['name'], e['object']

                if filter_lr:
                    if (rel_name == 'to the left of') or (rel_name == 'to the right of'):
                        continue

                # dataset quality check
                assert rel_obj_id in img_obj_dict

                if filter_set is not None:
                    should_proceed = (rel_name in filter_set) or (img_obj_dict[rel_obj_id]['name'] in filter_set)
                    if not should_proceed:
                        continue

                if rel_name in rel_dict:
                    rel_dict[rel_name].append([sub_id, rel_obj_id])
                else:
                    rel_dict[rel_name] = [[sub_id, rel_obj_id]]

        return rel_dict

    def gen_img_boxes(self, img_id):

        sGraph = self.get_sGraph(img_id)
        img_obj_dict = sGraph['objects']

        box_ls, caption_ls = [], []
        for obj_id, obj_info in img_obj_dict.items():

            obj_name = obj_info['name']
            obj_attr = 'attr: %s' % ' '.join(obj_info['attributes'])
            obj_rel_ls = obj_info['relations']
            obj_rel = ' '.join(['%s(%s,%s)' % (e['name'], obj_name, img_obj_dict[e['object']]['name'])
                                for e in obj_rel_ls])

            box = [obj_info['x'], obj_info['y'], obj_info['w'], obj_info['h']]

            box_ls.append(box)
            caption_ls.append('%s\n%s\n%s' % (obj_name, obj_attr, obj_rel))

        return box_ls, caption_ls

    def load_img(self, img_id):
        img_path = joinpath(self.img_dir, img_id + '.jpg')
        assert os.path.isfile(img_path), '%s not exists' % img_path
        img = mpimg.imread(img_path).copy()

        return img

    def check(self):

        img_set = set([fn[:-4] for fn in os.listdir(self.img_dir)])
        sGraph_set = set(list(self.all_sGraph.keys()))
        train_set = set(list(self.train_sGraph.keys()))
        valid_set = set(list(self.val_sGraph.keys()))

        intersect = img_set.intersection(sGraph_set)

        print('img set', len(img_set))
        print('sg set', len(sGraph_set))
        print('intersect', len(intersect))
        print('train', len(train_set))
        print('valid', len(valid_set))
        print('train_valid_inter', len(train_set.intersection(valid_set)))


def prep_car_data(data_root = '../../../dataset/gqa/scene_graph'):
    output_path = '../data/gqa'

    fact_domain_path = joinpath(output_path, 'fact_domains')
    valid_domain_path = joinpath(output_path, 'valid_domains')
    test_domain_path = joinpath(output_path, 'test_domains')

    os.mkdir(fact_domain_path)
    os.mkdir(valid_domain_path)
    os.mkdir(test_domain_path)

    gqa = GQADataset(data_root)
    filter_under = 1500
    un_mergDict, rel_mergDict = {}, {}
    un_filterSet, rel_filterSet = set(), set()

    # with open('car_img.txt') as f:
    #     img_id_ls = [line.strip() for line in f]

    with open('un_merge.txt') as f:
        for line in f:
            merge_name, parts = line.split(': ')
            names = parts.strip().split(',')
            for name in names:
                un_mergDict[name] = merge_name
    with open('rel_merge.txt') as f:
        for line in f:
            merge_name, parts = line.split(': ')
            names = parts.strip().split(',')
            for name in names:
                rel_mergDict[name] = merge_name

    freq_dict = {}
    # with open('car_un_freq.txt') as f:
    with open('obj_freq.txt') as f:
        for line in f:
            parts = line.strip().split(' ')
            freq = int(parts[-1])
            name = ' '.join(parts[:-1])
            if name in un_mergDict:
                name = un_mergDict[name]

            if name in freq_dict:
                freq_dict[name] += freq
            else:
                freq_dict[name] = freq
    un_filterSet.update([k for k, v in freq_dict.items() if v < filter_under])

    freq_dict = {}
    # with open('car_rel_freq.txt') as f:
    with open('rel_freq.txt') as f:
        for line in f:
            parts = line.strip().split(' ')
            freq = int(parts[-1])
            name = ' '.join(parts[:-1])
            if name in rel_mergDict:
                name = rel_mergDict[name]

            if name in freq_dict:
                freq_dict[name] += freq
            else:
                freq_dict[name] = freq
    rel_filterSet.update([k for k, v in freq_dict.items() if v < filter_under])

    # 8/1/1 split
    num_domains = len(gqa.all_sGraph)
    valid_ind = math.ceil(num_domains * 0.8)
    test_ind = math.ceil(num_domains * 0.9)
    domain_path_ls = [fact_domain_path, valid_domain_path, test_domain_path]

    un_pred_set, bi_pred_set = set(), set()
    cur_ind = 0
    # ht_dict, th_dict = {}, {}
    for img_id, _ in tqdm(gqa.all_sGraph.items()):

        domain_path = 0 if cur_ind < valid_ind else 1
        domain_path = domain_path if cur_ind < test_ind else 2
        domain_path = domain_path_ls[domain_path]


        sgraph = gqa.get_sGraph(img_id)
        objs_dict = sgraph['objects']
        rel_dict = gqa.get_flatRel(img_id, filter_lr=True)
        fact_set = set()

        for rel_name, sub_obj_id_ls in rel_dict.items():
            if rel_name in rel_filterSet:
                continue

            for sub_id, obj_id in sub_obj_id_ls:
                sub_name, obj_name = objs_dict[sub_id]['name'], objs_dict[obj_id]['name']
                sub_name = un_mergDict[sub_name] if sub_name in un_mergDict else sub_name
                obj_name = un_mergDict[obj_name] if obj_name in un_mergDict else obj_name
                if (sub_name in un_filterSet) or (obj_name in un_filterSet):
                    continue

                rel_name = rel_mergDict[rel_name] if rel_name in rel_mergDict else rel_name

                rel_name = rel_name.replace(' ', '_')
                sub_name = sub_name.replace(' ', '_')
                obj_name = obj_name.replace(' ', '_')

                un_pred_set.add('%s(type)' % sub_name)
                un_pred_set.add('%s(type)' % obj_name)
                bi_pred_set.add('%s(type,type)' % rel_name)

                fact_set.add('1\t%s(%s)' % (sub_name, sub_id))
                fact_set.add('1\t%s(%s)' % (obj_name, obj_id))
                fact_set.add('1\t%s(%s,%s)' % (rel_name, sub_id, obj_id))

        if len(fact_set) > 0:

            with open(joinpath(domain_path, img_id), 'w') as f:
                for fact in fact_set:
                    f.write('%s\n' % fact)

        cur_ind += 1

    for pn in list(un_pred_set) + list(bi_pred_set):
        with open(joinpath(output_path, 'pred.txt'), 'a') as f:
            f.write('%s\n' % pn)



if __name__ == '__main__':

    random.seed(10)

    prep_car_data(sys.argv[1])
