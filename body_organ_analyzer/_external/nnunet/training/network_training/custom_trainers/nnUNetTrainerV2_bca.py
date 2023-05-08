from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_bca(nnUNetTrainerV2):
    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            use_ignore_label=255,
            use_adamw=False,
        )

        self.max_num_epochs = 4000
