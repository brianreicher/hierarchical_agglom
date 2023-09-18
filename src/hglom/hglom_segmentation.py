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
        mask_file: Optional[str] = None,
        mask_dataset: Optional[str] = None,
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
        thresholds: list[float] = [0.66, 0.68, 0.70]
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
        self.mask_file: str = mask_file
        self.mask_dataset: str = mask_dataset

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
            mask_file=self.mask_file,
            mask_dataset=self.mask_dataset,
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
