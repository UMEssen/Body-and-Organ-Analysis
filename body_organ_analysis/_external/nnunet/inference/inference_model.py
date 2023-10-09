import os
from batchgenerators.augmentations.utils import pad_nd_image

from nnunet.network_architecture.neural_network import SegmentationNetwork
import tritonclient.grpc as grpcclient
import torch
import numpy as np
from typing import Tuple, Union, Optional, List
from functools import partial
import queue
import logging

logger = logging.getLogger(__name__)


class TritonNetwork(SegmentationNetwork):
    def __init__(self, model_name: str, model_versions: List[str], headers: dict):
        super(SegmentationNetwork, self).__init__()
        self.client = grpcclient.InferenceServerClient(url=os.environ["TRITON_URL"])
        self.model_name = model_name
        self.model_versions = model_versions
        self.version_iter = -1
        self.headers = headers
        model_metadata = self.client.get_model_metadata(model_name, headers=headers)
        input_shape = [1] + model_metadata.inputs[0].shape[1:]
        self.input_tensor = grpcclient.InferInput(
            "input", input_shape, model_metadata.inputs[0].datatype
        )
        # 8 had 458.15s, 4 had 452, 3 had 443.66s, 2 had 510.10s
        self.max_parallel_requests = 3
        self.num_current_requests = 0
        self.response_queue = queue.Queue()  # type: ignore

    def update_version_iter(self):
        logger.debug(f"Updating version iteration to {self.version_iter + 1}")
        self.version_iter += 1

    def reset_version_iter(self):
        logger.debug("Resetting version iteration to -1.")
        self.version_iter = -1

    def _internal_predict_3D_3Dconv_tiled(
        self,
        x: np.ndarray,
        step_size: float,
        do_mirroring: bool,
        mirror_axes: tuple,
        patch_size: tuple,
        regions_class_order: tuple,
        use_gaussian: bool,
        pad_border_mode: str,
        pad_kwargs: dict,
        all_in_gpu: bool,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose:
            print("step_size:", step_size)
        if do_mirroring:
            raise NotImplementedError(
                "The version with mirroring has only been implemented partially. "
                "Read the comments on how to implement it."
            )

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(
            x, patch_size, pad_border_mode, pad_kwargs, True, None
        )
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(
            patch_size, data_shape[1:], step_size
        )
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        # TODO: tiles > 1 was removed for simplicity otherwise we
        #  have to figure out another way to remove the gaussian layer from the model
        if use_gaussian:  # TODO: and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]
            ):
                if verbose:
                    print("computing Gaussian")
                gaussian_importance_map = self._get_gaussian(
                    patch_size, sigma_scale=1.0 / 8
                )

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose:
                    print("done")
            add_for_nb_of_preds = self._gaussian_3d
        else:
            raise NotImplementedError("For the time being we always use gaussians!")

        aggregated_results = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32
        )
        aggregated_nb_of_predictions = np.zeros(
            [self.num_classes] + list(data.shape[1:]), dtype=np.float32
        )

        # TODO: For mirroring, add a mirror idx to the iterator
        patch_iter = iter(
            [
                (
                    slice(x, x + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(z, z + patch_size[2]),
                )
                for x in steps[0]
                for y in steps[1]
                for z in steps[2]
            ]
        )

        while patch_iter is not None or self.num_current_requests > 0:
            if (
                patch_iter is not None
                and self.num_current_requests < self.max_parallel_requests
            ):
                try:
                    slice_indices = next(patch_iter)
                    self.input_tensor.set_data_from_numpy(
                        data[
                            None,
                            :,
                            slice_indices[0],
                            slice_indices[1],
                            slice_indices[2],
                        ],
                    )
                    # TODO: For mirroring, send both the indices and the iterator
                    self.client.async_infer(
                        model_name=self.model_name,
                        model_version=self.model_versions[self.version_iter],
                        inputs=[self.input_tensor],
                        headers=self.headers,
                        callback=partial(self._async_callback, slice_indices),  # type: ignore
                    )
                    self.num_current_requests += 1
                    continue
                except StopIteration:
                    patch_iter = None
            # Gather one response
            success, response = self.response_queue.get()
            if not success:
                raise response
            if response is not None:
                # TODO: For mirroring, store the result in another array first until
                #  all mirror ids have been retrieved
                slicer_idx, predicted_patch = response
                aggregated_results[
                    :, slicer_idx[0], slicer_idx[1], slicer_idx[2]
                ] += predicted_patch[0]
                aggregated_nb_of_predictions[
                    :, slicer_idx[0], slicer_idx[1], slicer_idx[2]
                ] += add_for_nb_of_preds
            self.num_current_requests -= 1

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [
                slice(0, aggregated_results.shape[i])
                for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
            ]
            + slicer[1:]
        )
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            predicted_segmentation = np.zeros(
                aggregated_results.shape[1:], dtype=np.float32
            )
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[aggregated_results[i] > 0.5] = c

        if verbose:
            print("prediction done")
        return predicted_segmentation, aggregated_results

    @staticmethod
    def mirror_info(x: np.ndarray, mirror_idx: int, mirror_axes):
        if mirror_idx == 0:
            return x
        if mirror_idx == 1 and (2 in mirror_axes):
            return np.flip(x, (4,))
        if mirror_idx == 2 and (1 in mirror_axes):
            return np.flip(x, (3,))
        if mirror_idx == 3 and (2 in mirror_axes) and (1 in mirror_axes):
            return np.flip(x, (4, 3))
        if mirror_idx == 4 and (0 in mirror_axes):
            return np.flip(x, (2,))
        if mirror_idx == 5 and (0 in mirror_axes) and (2 in mirror_axes):
            return np.flip(x, (4, 2))
        if mirror_idx == 6 and (0 in mirror_axes) and (1 in mirror_axes):
            return np.flip(x, (3, 2))
        if (
            mirror_idx == 7
            and (0 in mirror_axes)
            and (1 in mirror_axes)
            and (2 in mirror_axes)
        ):
            return np.flip(x, (4, 3, 2))

    def _internal_predict_3D_3Dconv(self, *args, **kwargs):
        raise NotImplementedError(
            "This function has not been implemented for triton yet."
        )

    def _internal_predict_3D_2Dconv_tiled(self, *args, **kwargs):
        raise NotImplementedError(
            "This function has not been implemented for triton yet."
        )

    def _internal_predict_3D_2Dconv(self, *args, **kwargs):
        raise NotImplementedError(
            "This function has not been implemented for triton yet."
        )

    def _internal_maybe_mirror_and_pred_3D(
        self,
        x: Union[np.ndarray, torch.tensor],
        mirror_axes: tuple,
        do_mirroring: bool = True,
        mult: np.ndarray or torch.tensor = None,
    ) -> None:
        raise NotImplementedError("This function is not used in triton mode.")

    def _async_callback(
        self,
        slicer_idx: int,
        result: grpcclient.InferResult,
        error: Optional[grpcclient.InferenceServerException],
    ):
        if error:
            self.response_queue.put((False, error))
            return

        assert len(result._result.outputs) == 1
        self.response_queue.put((True, (slicer_idx, result.as_numpy("output"))))
