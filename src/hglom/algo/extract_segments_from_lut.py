import daisy
import json
import logging
import numpy as np
import os
import time
from funlib.geometry import Coordinate, Roi
from funlib.segment.arrays import replace_values
from funlib.persistence import Array


logging.getLogger().setLevel(logging.INFO)


def extract_segmentation(
    fragments_file: str,
    fragments_dataset: str,
    merge_function: str,
    thresholds: list[float] = [0.66, 0.68, 0.70],
    block_size: list = [66, 66, 66],
    threshold: float = 0.48,
    num_workers: int = 7,
) -> bool:
    """Generate segmentation based on fragments using specified merge function.

    Args:
        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        merge_function (``str``):
            The method to use for merging fragments (e.g., 'hist_quant_75').

        thresholds (``list[float]``, optional):
            List of thresholds for segmentation. Default is [0.66, 0.68, 0.70].

        block_size (list of int, optional):
            The size of one block in world units (must be a multiple of voxel size). Default is [66, 66, 66].

        threshold (``float``, optional):
            The threshold to use for generating the segmentation. Default is 0.48.

        num_workers (``int``, optional):
            Number of workers to use when reading the region adjacency graph blockwise. Default is 7.

    Returns:
        ``bool``:
            True if segmentation generation was successful, False otherwise.
    """

    results_file: str = os.path.join(fragments_file, "results.json")

    lut_dir: str = os.path.join(fragments_file, "luts", "fragment_segment")

    fragments: Array = daisy.open_ds(filename=fragments_file, ds_name=fragments_dataset)

    total_roi: Roi = fragments.roi
    read_roi = daisy.Roi(offset=(0,) * 3, shape=Coordinate(block_size))
    write_roi: Roi = read_roi
    voxel_size: Coordinate = fragments.voxel_size

    logging.info(msg="Preparing segmentation dataset...")

    if os.path.exists(path=results_file):
        with open(file=results_file, mode="r") as f:
            results = json.load(f)
            bests: list = [results[x]["best_voi"]["threshold"] for x in results.keys()]
            for best in bests:
                if best not in thresholds:
                    thresholds.append(best)

    for threshold in thresholds:
        seg_name: str = f"segmentation_{threshold}"

        try:
            lut_filename: str = f"seg_hglom_edges_{merge_function}_{int(threshold*100)}"
            os.path.join(lut_dir, lut_filename + ".npz")
        except:
            continue

        start: float = time.time()
        logging.info(fragments.roi)
        logging.info(fragments.voxel_size)
        segmentation = daisy.prepare_ds(
            filename=fragments_file,
            ds_name=seg_name,
            total_roi=total_roi,
            voxel_size=voxel_size,
            dtype=np.uint64,
            write_roi=write_roi,
            delete=True,
        )

        lut_filename: str = f"seg_hglom_edges_{merge_function}_{int(threshold*100)}"

        lut: str = os.path.join(lut_dir, lut_filename + ".npz")

        assert os.path.exists(path=lut), f"{lut} does not exist"

        logging.info(msg="Reading fragment-segment LUT...")

        lut = np.load(file=lut)["fragment_segment_lut"]

        logging.info(msg=f"Found {len(lut[0])} fragments in LUT")

        num_segments: int = len(np.unique(ar=lut[1]))
        logging.info(msg=f"Relabelling fragments to {num_segments} segments")

        task = daisy.Task(
            task_id="ExtractSegmentsTask",
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: segment_in_block(
                block=b, segmentation=segmentation, fragments=fragments, lut=lut
            ),
            fit="shrink",
            num_workers=num_workers,
        )

        done: bool = daisy.run_blockwise(tasks=[task])

        if not done:
            raise RuntimeError(
                "Extraction of segmentation from LUT failed for (at least) one block"
            )

        logging.info(
            msg=f"Took {time.time() - start} seconds to extract segmentation from LUT"
        )

    return True


def segment_in_block(block, segmentation, fragments, lut) -> None:
    logging.info(msg="Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.write_roi)

    # replace values, write to empty array
    relabelled: np.ndarray = np.zeros_like(fragments)
    relabelled: np.ndarray = replace_values(
        in_array=fragments, old_values=lut[0], new_values=lut[1], out_array=relabelled
    )

    segmentation[block.write_roi] = relabelled
