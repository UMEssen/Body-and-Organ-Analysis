from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import (
    nnUNetTrainerV2CascadeFullRes,
)


class nnUNetTrainerV2CascadeFullRes_ep250_nomirror(nnUNetTrainerV2CascadeFullRes):
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
        previous_trainer="nnUNetTrainerV2_ep250_nomirror",
        fp16=False,
    ):
        previous_trainer = "nnUNetTrainerV2_ep250_nomirror"
        super().__init__(
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            previous_trainer,
            fp16,
        )

        self.max_num_epochs = 250
