from typing import Optional
import os
import numpy as np
from funlib.geometry import Coordinate
from .algo.extract_fragments import extract_fragments
from .algo.agglomerate_blockwise import agglomerate
from .algo.find_segments import find_segments
from .algo.extract_segments_from_lut import extract_segmentation
from .utils import neighborhood
import time


class PostProcessor:
    """Driver for post-processing segmentation.

    Args:
        affs_file (str):
            Path (relative or absolute) to the zarr file containing predicted affinities to generate fragments for.

        affs_dataset (str):
            The name of the affinities dataset in the affs_file to read from.

        context (daisy.Coordinate, optional):
            A coordinate object (3-dimensional) denoting how much contextual space to grow for the total volume ROI.
            Defaults to a Coordinate of the maximum absolute value of the neighborhood if ``None``.

        sample_name (str, optional):
            A string containing the sample name (run name of the experiment) to denote for the MongoDB collection_name.
            Default is None.

        fragments_file (str, optional):
            Path (relative or absolute) to the zarr file to write fragments to.
            Default is "", which sets the file to the same as the given ``affs_file``.

        fragments_dataset (str, optional):
            The name of the fragments dataset to read/write to in the fragments_file.
            Default is "frags."

        seg_file (str, optional):
            Path (relative or absolute) to the zarr file to write fragments to.
            Default is "", which sets the file to the same as the given ``affs_file``.

        seg_dataset (str, optional):
            The name of the segmentation dataset to write to.
            Default is "seg."

        seeds_file (str, optional):
            Path (relative or absolute) to the zarr file containing seeds.
            Default is None.

        seeds_dataset (str, optional):
            The name of the seeds dataset in the seeds file to read from.
            Default is None.
        
        filter_val (float, optional):
            The amount for which fragments will be filtered if their average falls below said value.
            Default is 0.5.

        neighborhood_length (int, optional):
            Number of neighborhood offsets to use, default is 12. See ``utils.py`` for full neighborhood.

        nworkers_frags (int, optional):
            Number of distributed workers to run the Daisy parallel fragment task with.
            Default is 10.

        merge_function (str, optional):
            Name of the segmentation algorithm used to denote in the MongoDB edge collection.
            Default is "hist_quant_75".

        epsilon_agglomerate (float, optional):
            A threshold parameter for agglomeration. Default is 0.05.
        
        nworkers_agglom (int, optional):
            Number of distributed workers to run the Daisy parallel supervoxel task with.
            Default is 7.
        
        thresholds_minmax (list, optional):
            The lower and upper bounds to use for generating thresholds. Default is [0, 1].

        thresholds_step (float, optional):
            The step size to use when generating thresholds between min/max. Default is 0.02.

        block_size (list, optional):
            The size of one block in world units (must be a multiple of voxel size). Default is [1056, 1056, 1056].

        lut_threshold (float, optional):
            The threshold to use for generating the segmentation. Default is 0.48.

        nworkers_lut (int, optional):
            Number of distributed workers to run the Daisy parallel LUT extraction task with.
            Default is 7.
        
        thresholds (list, optional):
            List of thresholds for segmentation. Default is [0.66, 0.68, 0.70].

    """""
    def __init__(
        self,
        affs_file: str,
        affs_dataset: str,
        context: Optional[Coordinate] = None,
        sample_name: Optional[str] = None,
        fragments_file: Optional[str] = "",
        fragments_dataset: Optional[str] = "frags",
        seg_file: Optional[str] = "",
        seg_dataset: Optional[str] = "seg",
        seeds_file: Optional[str] = None,
        seeds_dataset: Optional[str] = None,
        filter_val: Optional[float] = 0.5,
        neighborhood_length: Optional[int] = 12,
        nworkers_frags: Optional[int] = 20,
        merge_function: Optional[str] = "hist_quant_75",
        epsilon_agglomerate: Optional[float] = 0.05,
        nworkers_agglom: Optional[int] = 7,
        thresholds_minmax: Optional[list] = [0, 1],
        thresholds_step: Optional[float] = 0.02,
        block_size: Optional[list] = [1056, 1056, 1056],
        lut_threshold: Optional[float] = 0.48,
        nworkers_lut: Optional[int] = 7,
        thresholds: list[float] = [0.66, 0.68, 0.70],
    ) -> None:
        # set sample name
        self.sample_name: str = sample_name

        # dataset vars
        self.affs_file: str = affs_file
        self.affs_dataset: str = affs_dataset

        if fragments_file == "":
            self.fragments_file = affs_file
        else:
            self.fragments_file: str = fragments_file

        self.fragments_dataset: str = fragments_dataset
        self.seeds_file: str = seeds_file
        self.seeds_dataset: str = seeds_dataset

        if seg_file == "":
            self.seg_file = affs_file
        else:
            self.seg_file: str = seg_file

        self.seg_dataset: str = seg_dataset

        # dataset processing vars
        if context is not None:
            self.context: Coordinate = context
        else:
            self.context: Coordinate = Coordinate(
                np.max(a=np.abs(neighborhood[:neighborhood_length]), axis=0)
            )

        self.filter_val: float = filter_val
        self.merge_function: str = merge_function
        self.epsilon_agglomerate: float = epsilon_agglomerate
        self.thresholds_minmax: list = thresholds_minmax
        self.threholds_step: float = thresholds_step
        self.lut_threshold: float = lut_threshold
        self.thresholds: list[float] = thresholds

        # Daisy vars
        self.nworkers_frags: int = nworkers_frags
        self.nworkers_agglom: int = nworkers_agglom
        self.nworkers_lut: int = nworkers_lut
        self.block_size: list = block_size

    def segment(
        self,
    ) -> bool:
        if self.sample_name is None:
            self.sample_name: str = "htem" + str(
                hash(
                    f"FROM{os.path.join(self.affs_file, self.affs_dataset)}TO{os.path.join(self.fragments_file, self.fragments_dataset)}AT{time.strftime('%Y%m%d-%H%M%S')}".replace(
                        ".", "-"
                    ).replace(
                        "/", "-"
                    )
                )
            )
        success: bool = extract_fragments(
            affs_file=self.affs_file,
            affs_dataset=self.affs_dataset,
            fragments_file=self.fragments_file,
            fragments_dataset=self.fragments_dataset,
            seeds_file=self.seeds_file,
            seeds_dataset=self.seeds_dataset,
            num_workers=self.nworkers_frags,
            context=self.context,
            filter_fragments=self.filter_val,
            epsilon_agglomerate=self.epsilon_agglomerate,
            merge_function=self.merge_function,
        )

        if success:
            success: bool = agglomerate(
                affs_file=self.affs_file,
                affs_dataset=self.affs_dataset,
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                context=self.context,
                num_workers=self.nworkers_agglom,
                merge_function=self.merge_function,
            )

        if success:
            success: bool = find_segments(
                affs_file=self.affs_file,
                affs_dataset=self.affs_dataset,
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                thresholds_minmax=self.thresholds_minmax,
                thresholds_step=self.threholds_step,
                merge_function=self.merge_function,
            )

        if success:
            success: bool = extract_segmentation(
                fragments_file=self.fragments_file,
                fragments_dataset=self.fragments_dataset,
                merge_function=self.merge_function,
                thresholds=self.thresholds,
                block_size=self.block_size,
                threshold=self.lut_threshold,
                num_workers=self.nworkers_lut,
            )
        return success
