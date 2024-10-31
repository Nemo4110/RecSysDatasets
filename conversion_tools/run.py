# @Time   : 2020/9/18
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


import argparse
import importlib

from src.utils import dataset2class, click_dataset, multiple_dataset, multiple_item_features


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--interaction_type', type=str, default=None)
    parser.add_argument('--duplicate_removal', action='store_true')

    parser.add_argument("--do_split", action='store_true', default=False)
    parser.add_argument("--do_seq_rec", action='store_true', default=False)
    parser.add_argument("--MAX_ITEM_LIST_LENGTH", type=int, default=20)

    parser.add_argument('--item_feature_name', type=str, default='none')

    parser.add_argument('--convert_inter', action='store_true')
    parser.add_argument('--convert_item', action='store_true')
    parser.add_argument('--convert_user', action='store_true')

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'



    input_args = [args.input_path, args.output_path]
    dataset_class_name = dataset2class[args.dataset.lower()]
    dataset_class = getattr(importlib.import_module('src.extended_dataset'), dataset_class_name)
    if dataset_class_name in multiple_dataset:
        input_args.append(args.interaction_type)
    if dataset_class_name in click_dataset:
        input_args.append(args.duplicate_removal)
    if dataset_class_name in multiple_item_features:
        input_args.append(args.item_feature_name)
    if "MIMICIIIDrugDataset" == dataset_class_name or "MIMICIIILabItemDataset" == dataset_class_name:
        input_args.append(args.do_split)
        input_args.append(args.do_seq_rec)

        # labitem推荐时要调成10，否则32G内存的服务器运行recbole初始化seq数据集时会爆内存
        input_args.append(args.MAX_ITEM_LIST_LENGTH)

    datasets = dataset_class(*input_args)

    if args.convert_item:
        datasets.convert_item()
    if args.convert_user:
        datasets.convert_user()
    if args.convert_inter:
        datasets.convert_inter()
