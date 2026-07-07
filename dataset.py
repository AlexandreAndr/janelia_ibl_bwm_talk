import os
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import numpy as np
from torch_brain.data import Data
from torch_brain.dataset import Dataset, DatasetIndex, SpikingDatasetMixin

Split: TypeAlias = Literal["train", "val", "test"]
BrainRegionAtlas: TypeAlias = Literal["allen", "beryl", "cosmos"]


class IBLBrainWideMap2025(SpikingDatasetMixin, Dataset):
    """
    Dataset for the IBL Brain-Wide Map (2025).

    Args:
        root: The root directory of the dataset.
        dirname: The name of the dataset (and the directory containing its data).
        recording_ids: The recording ids to include in the dataset. If None, all
                       recordings in the dataset are included.
        transform: The transform(s) to apply to the data.
        split: The split of the dataset (train, val, test).
    """

    spiking_dataset_mixin_uniquify_unit_ids = True

    # TODO define total context duration
    CONTEXT_WINDOW = 1.0  # seconds
    BEHAVIOR_SFREQ = 50  # Hz

    def __init__(
        self,
        root: str,
        dirname: str = "ibl_brain_wide_map_2025",
        recording_ids: str | list[str] | None = None,
        transform: Any | None = None,
        split: Split | None = "train",
    ):
        recording_ids_path = os.path.join(os.path.dirname(__file__), "data", "recording_ids.txt")

        with open(recording_ids_path) as f:
            all_recording_ids = [line.strip() for line in f if line.strip()]

        self.dataset_dir = Path(root) / dirname

        if isinstance(recording_ids, str):
            recording_ids = [recording_ids]

        if recording_ids is None or len(recording_ids) == 0:
            recording_ids = all_recording_ids

        invalid = set(recording_ids) - set(all_recording_ids)
        assert not invalid, f"{invalid} not found in recording ids"

        super().__init__(
            dataset_dir=self.dataset_dir,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=[
                "session.id",
                "subject.id",
                "units.id",
                "probes.id",
            ],
        )

        if split is not None and split not in get_args(Split):
            raise ValueError(f"split={split} not well define should be ('train', 'val', 'test')")

        self.split = split
        self.return_target = False

    def __getitem__(self, index: DatasetIndex):
        data = self.get_recording(index.recording_id, "")
        sample = data.slice(index.start, index.end)
        sample = self.dataset_transform(sample)

        if self.return_target:
            target = self._get_target(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_target:
            return sample, target

        return sample

    def dataset_transform(self, data: Data) -> Data:
        """
        Defines dataset-level transformations that are applied to all recordings in the Dataset.
        This method can be applied on an entire recording or a single slice.

        Args:
            data: The Data object to apply the transformations to.

        Returns:
            The Data object with the transformations applied.

        Example:
            >>> dataset = IBLBrainWideMap2025(...)
            >>> data = dataset.get_recording(...)
            >>> data = dataset.dataset_transform(data)
            >>> slice = data.slice(0.5, 1.5)
            >>> slice = dataset.dataset_transform(slice)
        """

        return data

    def _get_target(self, data: Data):
        raise NotImplementedError

    def get_sampling_intervals(self):
        raise NotImplementedError(
            "get_sampling_intervals is too general for the base brain wide map dataset, "
            "look at the task suite level or create a more specific."
        )

    def get_session_ids(self) -> list[str]:
        ans = [self.get_recording(rid).session.id for rid in self.recording_ids]
        return np.sort(np.unique(ans)).tolist()

    def get_subject_ids(self) -> list[str]:
        ans = [self.get_recording(rid).subject.id for rid in self.recording_ids]
        return np.sort(np.unique(ans)).tolist()

    def get_num_unit_per_recording(self):
        return {rid: len(self.get_recording(rid).units) for rid in self.recording_ids}

    def get_brain_regions(self, atlas="cosmos") -> list[str]:
        """Return a sorted list of all brain regions across all recordings in the dataset."""

        assert atlas in get_args(BrainRegionAtlas), f"{atlas} is not a valid atlas."

        ans = [
            self.get_recording(rid).get_nested_attribute(f"units.region_{atlas}")
            for rid in self.recording_ids
        ]
        return np.sort(np.unique(np.concatenate(ans))).tolist()
