from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
import os
import json
import numpy as np
import pandas as pd


@dataclass
class Params:
    decoder_name: Optional[str] = None
    encoder_name: Optional[str] = None
    current_time: Optional[str] = None
    benchmark: Optional[str] = None
    MultiModel: Optional[str] = None
    order: Optional[int] = None
    metric: Optional[str] = None
    average__a_0_t: Optional[float] = None
    tasks: Optional[str] = None


def main():
    parser = HfArgumentParser(Params)
    params = parser.parse_args_into_dataclasses()[0]

    task_list = params.tasks.split(",")
    task_dict = {k: v for k, v in (item.split(':', 1) for item in task_list)}
    task_order_list = list(task_dict)
    sequence_and_task_list = [str(i + 1).zfill(2) + "-" + task_order_list[i] for i in range(len(task_order_list))]

    logs_and_outputs_path = os.path.join("logs_and_outputs",
                                         str(params.benchmark) + "+" + str(params.decoder_name) + "+" + str(
                                             params.encoder_name) + "+" + str(params.MultiModel),
                                         str(params.current_time) + "+" + str(params.order))

    metrics_list = []

    for sequence_number, sequence_and_task in enumerate(sequence_and_task_list, start=1):
        inner_metrics_list = []
        inner_sequence_and_task_list = sequence_and_task_list[:sequence_number]

        for inner_sequence_and_task in inner_sequence_and_task_list:
            if len(inner_sequence_and_task_list) == 1:
                metrics_path = os.path.join("LoRA_model", str(params.benchmark) + "+" + str(params.decoder_name),
                                            inner_sequence_and_task.split("-")[-1], "metrics.json")
            else:
                metrics_path = os.path.join(logs_and_outputs_path, sequence_and_task, "tasks_metric",
                                            inner_sequence_and_task, "metrics.json")

            with open(metrics_path, 'r', encoding='utf-8') as metrics_file:
                inner_metrics_list.append(json.load(metrics_file)[params.metric])

        metrics_list.append(inner_metrics_list)

    n = len(metrics_list)

    mat = np.full((n, n), np.nan, dtype=float)

    for i, row in enumerate(metrics_list):
        mat[i, :i + 1] = row

    df = pd.DataFrame(mat, index=sequence_and_task_list, columns=sequence_and_task_list)

    df.to_excel(os.path.join(logs_and_outputs_path, "mat_" + params.metric + ".xlsx"), index=True, header=True,
                index_label="tasks")

    current_task_metrics_dict = {sequence_and_task_list[i]: mat[i, i] for i in range(len(sequence_and_task_list))}

    with open(os.path.join(logs_and_outputs_path, "diag_" + params.metric + ".json"), "w", encoding="utf-8") as f:
        json.dump(current_task_metrics_dict, f, ensure_ascii=False, indent=4)

    # Average Performance (AP)
    with open(os.path.join(logs_and_outputs_path, sequence_and_task_list[-1], "metrics.json"), 'r',
              encoding='utf-8') as metrics_file:
        AP = json.load(metrics_file)["test_" + params.metric]

    # Forgetting Rate (FR)

    col_max = np.nanmax(mat[:-1], axis=0)[:-1]

    FR = np.nanmean(col_max - mat[-1][:-1])

    # Forward Transfer (FT)
    vals = np.array(list(current_task_metrics_dict.values()), dtype=float)
    FT = np.nanmean(vals) - params.average__a_0_t

    # Backward Transfer (BT)
    BT = np.nanmean(mat[-1][:-1] - vals[:-1])

    # summary
    summary = {
        "AP": AP,
        "FR": FR,
        "FT": FT,
        "BT": BT
    }

    with open(os.path.join(logs_and_outputs_path, "summary_" + params.metric + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
