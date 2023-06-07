import os
from pathlib import Path

import numpy as np

from totalsegmentator.libs import download_pretrained_weights
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.network_trainer import SegmentationNetwork
import torch.onnx
import torch.nn.functional as F


def create_config(name: str, input_dimension: str, output_dimension: str):
    info = [
        f'name: "{name}"',
        'platform: "onnxruntime_onnx"',
        "max_batch_size: 1",
        "",
        "input [",
        "    {",
        '        name: "input"',
        "        data_type: TYPE_FP32",
        f"        dims: {input_dimension}",
        "    }",
        "]",
        "",
        "output [",
        "    {",
        '        name: "output"',
        "        data_type: TYPE_FP32",
        f"        dims: {output_dimension}",
        "    }",
        "]",
        "",
        "version_policy: {",
        "   all { }",
        "}",
        "",
        "instance_group [",
        "    {",
        "      gpus: 1",
        "      kind: KIND_GPU",
        "      rate_limiter {",
        "          resources: [",
        "              {",
        '                  name: "gpu_lock"',
        "                  count: 1",
        "              }",
        "          ]",
        "       }",
        "    }",
        "]",
        "",
        'parameters { key: "enable_mem_arena" value: { string_value: "0" } }',
        'parameters { key: "enable_mem_pattern" value: { string_value: "0" } }',
        'parameters { key: "arena_extend_strategy" value: { string_value: "1" } }',
        'parameters { key: "memory.enable_memory_arena_shrinkage" '
        'value: { string_value: "gpu:1" } }',
        # 'parameters { key: "gpu_mem_limit" value: { string_value: "17179869184" } }',
        "",
    ]
    return "\n".join(info) + "\n"


class NetworkWithSoftmax(torch.nn.Module):
    def __init__(self, network: SegmentationNetwork, patch_size: np.ndarray):
        super().__init__()
        self.network = network
        gaussian_importance_map = network._get_gaussian(patch_size, sigma_scale=1.0 / 8)
        self.gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(
            self.network.get_device(), non_blocking=True
        )

    def forward(self, x):
        return F.softmax(self.network(x), 1) * self.gaussian_importance_map


task_name_for_id = {542: "Task542_BCA_inference"}

if __name__ == "__main__":
    """
    Download all pretrained weights
    """
    output_onxx = Path(os.environ["ONNX_OUTPUT"])
    export_anyway = True
    for task_id in [
        8,  # Task008_HepaticVessel
        150,  # Task150_icb_v0
        # 201, # Task201_covid # TODO: No link yet
        251,  # Task251_TotalSegmentator_part1_organs_1139subj
        252,  # Task252_TotalSegmentator_part2_vertebrae_1139subj
        253,  # Task253_TotalSegmentator_part3_cardiac_1139subj
        254,  # Task254_TotalSegmentator_part4_muscles_1139subj
        255,  # Task255_TotalSegmentator_part5_ribs_1139subj
        256,  # Task256_TotalSegmentator_3mm_1139subj
        258,  # Task258_lung_vessels_248subj
        260,  # Task260_hip_implant_71subj
        269,  # Task269_Body_extrem_6mm_1200subj
        273,  # Task273_Body_extrem_1259subj
        315,  # Task315_thoraxCT
        503,  # Task503_cardiac_motion
        542,  # Task542_BCA_inference
    ]:
        if task_id not in {542}:
            weights_path = download_pretrained_weights(task_id)
            task_name = weights_path.name
            folds = [0]
        else:
            task_name = task_name_for_id[task_id]
            weights_path = (
                Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
                / "nnUNet"
                / "3d_fullres"
                / task_name
            )
            folds = [0, 1, 2, 3, 4]
        plans_folder = [p for p in weights_path.glob("*") if p.is_dir()][0]
        trainer, params = load_model_and_checkpoint_files(
            str(plans_folder),
            folds,
            mixed_precision=True,
            checkpoint_name="model_final_checkpoint",
            prevent_triton=True,
        )
        for fold, param in zip(folds, params):
            output_folder = output_onxx / f"nnUNet_{task_name}" / f"{fold + 1}"
            model_name = output_folder / "model.onnx"
            if not model_name.exists() or export_anyway:
                trainer.load_checkpoint_ram(param, train=False)
                output_folder.mkdir(parents=True, exist_ok=True)
                net = trainer.network
                net.do_ds = False
                checkpoint = torch.load(
                    plans_folder / "fold_0" / "model_final_checkpoint.model"
                )
                net.load_state_dict(checkpoint["state_dict"])
                net.eval()

                batch_size = net.DEFAULT_BATCH_SIZE_3D
                input_shape = trainer.patch_size
                dummy_input = torch.randn(
                    batch_size, trainer.num_input_channels, *input_shape
                ).to("cuda")
                # Export the model
                torch.onnx.export(
                    NetworkWithSoftmax(net, trainer.patch_size),  # model being run
                    dummy_input,  # model input (or a tuple for multiple inputs)
                    str(model_name),
                    # where to save the model (can be a file or file-like object)
                    export_params=True,
                    # store the trained parameter weights inside the model file
                    opset_version=15,  # the ONNX version to export the model to
                    do_constant_folding=True,
                    # whether to execute constant folding for optimization
                    input_names=["input"],  # the model's input names
                    output_names=["output"],  # the model's output names
                    dynamic_axes={
                        "input": {0: "batch_size"},  # variable length axes
                        "output": {0: "batch_size"},
                    },
                )
                input_shape_onnx = [trainer.num_input_channels] + list(input_shape)
                output_shape_onnx = [trainer.num_classes] + list(input_shape)
                # print(input_shape_onnx)
                # print(output_shape_onnx)
                with (output_folder.parent / "config.pbtxt").open("w") as of:
                    of.write(
                        create_config(
                            name=output_folder.parent.name,
                            input_dimension=f"[{', '.join(map(str, input_shape_onnx))}]",
                            output_dimension=f"[{', '.join(map(str, output_shape_onnx))}]",
                        )
                    )
                # exit()
                continue
            else:
                print(f"The model {output_folder.parent.name} already exists.")
