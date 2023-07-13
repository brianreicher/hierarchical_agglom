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
    fragments_file,
    fragments_dataset,
    edges_collection,
    block_size: list = [64, 64, 64],
    threshold: float = 0.48,
    num_workers: int = 7,
) -> bool:
    """
    Args:
        fragments_file (``string``):
            Path to file (zarr/n5) containing fragments (supervoxels) and output segmentation.
        fragments_dataset (``string``):
            Name of fragments dataset (e.g `volumes/fragments`)
        edges_collection (``string``):
            The name of the MongoDB database edges collection to use.
        threshold (``float``):
            The threshold to use for generating a segmentation.
        block_size (``tuple`` of ``int``):
            The size of one block in world units (must be multiple of voxel
            size).
        out_dataset (``string``):
            Name of segmentation dataset (e.g `volumes/segmentation`).
        num_workers (``int``):
            How many workers to use when reading the region adjacency graph
            blockwise.
        roi_offset (array-like of ``int``, optional):
            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.
        roi_shape (array-like of ``int``, optional):
            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
        run_type (``string``, optional):
            Can be used to direct luts into directory (e.g testing, validation,
            etc).
    """

    results_file: str = os.path.join(fragments_file, "results.json")

    lut_dir: str = os.path.join(fragments_file, "luts", "fragment_segment")

    fragments: Array = daisy.open_ds(filename=fragments_file, ds_name=fragments_dataset)

    total_roi: Roi = fragments.roi
    read_roi = daisy.Roi(offset=(0,) * 3, shape=Coordinate(block_size))
    write_roi: Roi = read_roi

    logging.info(msg="Preparing segmentation dataset...")

    thresholds: list[float] = [0.64, 0.74, 0.84, 0.94]

    if os.path.exists(path=results_file):
        with open(file=results_file, mode="r") as f:
            results = json.load(f)
            bests: list = [results[x]["best_voi"]["threshold"] for x in results.keys()]
            for best in bests:
                if best not in thresholds:
                    thresholds.append(best)

    for threshold in thresholds:
        seg_name: str = f"segmentation_{threshold}"

        start: float = time.time()

        segmentation = daisy.prepare_ds(
            filename=fragments_file,
            ds_name=seg_name,
            total_roi=fragments.roi,
            voxel_size=fragments.voxel_size,
            dtype=np.uint64,
            write_roi=write_roi,
            delete=True,
        )

        lut_filename: str = f"seg_{edges_collection}_{int(threshold*100)}"

        lut: str = os.path.join(lut_dir, lut_filename + ".npz")

        assert os.path.exists(path=lut), f"{lut} does not exist"

        logging.info(msg="Reading fragment-segment LUT...")

        lut = np.load(file=lut)["fragment_segment_lut"]

        logging.info(msg=f"Found {len(lut[0])} fragments in LUT")

        num_segments: int = len(np.unique(ar=lut[1]))
        logging.info(msg=f"Relabelling fragments to {num_segments} segments")

        task = daisy.Task(
            task_id="ExtractSegmentationTask",
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
