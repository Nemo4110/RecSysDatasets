import pandas as pd
import os
import os.path as path
import argparse
import numpy as np

from tqdm import tqdm


def select_specified_day(dataset_path, dataset_name, benchmark_part, time_field, specified_day: int):
    df_inter = pd.read_csv(path.join(dataset_path, f"{dataset_name}.{benchmark_part}.inter"), sep='\t', header=0)
    return df_inter[df_inter[time_field] == specified_day]


def convert(input_data, selected_fields, output_file):
    output_data = pd.DataFrame()
    for column in selected_fields:
        output_data[column] = input_data.iloc[:, column]
    with open(output_file, 'w') as fp:
        fp.write('\t'.join([selected_fields[column] for column in output_data.columns]) + '\n')
        for i in tqdm(range(output_data.shape[0]), leave=False, ncols=100, ascii=True):
            fp.write('\t'.join([str(output_data.iloc[i, j])
                                for j in range(output_data.shape[1])]) + '\n')


def negative_sampling(df_inter, num_negatives=2):
    """
    对用户-物品交互记录进行负采样处理
    :param df_inter: 包含用户-物品交互记录的DataFrame
    :param num_negatives: 每个正样本对应的负样本数量
    :return: 处理后的DataFrame
    """
    df_inter = df_inter[['HADM_ID:token', 'NDC:token', 'TIMESTEP:float']]
    all_items = set(df_inter['NDC:token'].unique())

    results = []
    grouped = df_inter.groupby(['HADM_ID:token', 'TIMESTEP:float'])
    for (hadm_id, timestep), group in tqdm(grouped, ncols=100, ascii=True, total=grouped.ngroups, desc=f"negative sampling"):
        positive_items = set(group['NDC:token'])
        negative_items = list(all_items - positive_items)

        group_results = []  # 初始化当前组的结果列表
        for item in positive_items:
            group_results.append({'HADM_ID:token': hadm_id, 'NDC:token': item, 'label:float': 1})
        if negative_items:  # 确保有负样本可采样
            sampled_negatives = np.random.choice(negative_items,
                                                 size=min(num_negatives * len(positive_items), len(negative_items)),
                                                 replace=False)
            for neg_item in sampled_negatives:
                insert_pos = np.random.randint(0, len(group_results) + 1)
                group_results.insert(insert_pos, {'HADM_ID:token': hadm_id, 'NDC:token': neg_item, 'label:float': 0})

        results.extend(group_results)

    result_df = pd.DataFrame(results)
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dataset_path', type=str)
    parser.add_argument('--src_dataset_name', type=str)

    parser.add_argument('--tgt_dataset_path', type=str)
    parser.add_argument('--tgt_dataset_name', type=str)

    parser.add_argument('--benchmark_part', type=str)
    parser.add_argument('--time_field', type=str, default="TIMESTEP:float")
    parser.add_argument('--negative_sampling', action='store_true', default=False)

    parser.add_argument('--to_ctr', action='store_true', default=False)
    parser.add_argument('--to_seq', action='store_true', default=False)
    parser.add_argument('--to_gnn', action='store_true', default=False)

    args = parser.parse_args()

    os.makedirs(args.tgt_dataset_path, exist_ok=True)

    if args.to_gnn:
        df_inter = pd.read_csv(path.join(args.src_dataset_path, f"{args.src_dataset_name}.inter"), sep='\t', header=0)
        df_inter_day0 = df_inter[df_inter[args.time_field] == 0]
    else:
        df_inter_day0 = select_specified_day(args.src_dataset_path,
                                             args.src_dataset_name,
                                             args.benchmark_part,
                                             args.time_field,
                                             0)

    # negative sample
    if args.negative_sampling:
        df_inter_day0 = negative_sampling(df_inter_day0)

    if args.to_ctr:  # TO CTR
        selected_fields = {
            0: 'HADM_ID:token',
            1: 'NDC:token',
            2: 'label:float'
        }
    elif args.to_seq:  # TO SEQ
        selected_fields = {
            0: 'user_id:token',
            1: 'item_id:token',
            2: 'item_id_list:token_seq'
        }
    elif args.to_gnn:
        selected_fields = {
            0: 'user_id:token',
            1: 'item_id:token',
            2: 'DRUG_TYPE:token',
            3: 'PROD_STRENGTH:token',
            4: 'DOSE_VAL_RX:token',
            5: 'DOSE_UNIT_RX:token',
            6: 'FORM_VAL_DISP:token',
            7: 'FORM_UNIT_DISP:token',
            8: 'ROUTE:token',
            9: 'ROW_ID:float'
        }
    else:
        selected_fields = {
            0: 'HADM_ID:token',
            1: 'NDC:token',
        }

    df_inter_day0 = df_inter_day0[selected_fields.values()]
    if args.to_gnn:
        convert(df_inter_day0, selected_fields,
                path.join(args.tgt_dataset_path, f"{args.tgt_dataset_name}.inter"))
    else:
        convert(df_inter_day0, selected_fields,
                path.join(args.tgt_dataset_path, f"{args.tgt_dataset_name}.{args.benchmark_part}.inter"))