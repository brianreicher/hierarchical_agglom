import logging
import numpy as np
import os
import daisy
import time
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, graphs, Array
from lsd.post import watershed_in_block
from .utils import neighborhood

logging.getLogger().setLevel(logging.INFO)


def extract_fragments(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    seeds_file: str,
    seeds_dataset: str,
    context: tuple,
    num_workers: int = 20,
    fragments_in_xy=False,
    epsilon_agglomerate=0.05,
    mask_file=None,
    mask_dataset=None,
    filter_fragments=0.10,
    replace_sections=None,
    merge_function: str = "watershed",
) -> bool:
    """Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.

        num_workers (``int``):
            How many blocks to run in parallel.
    """
    start: float = time.time()
    logging.info(msg=f"Reading {affs_dataset} from {affs_file}")
    affs_ds: Array = daisy.open_ds(filename=affs_file, ds_name=affs_dataset, mode="r")

    seeds_ds: Array = open_ds(filename=seeds_file, ds_name=seeds_dataset)
    voxel_size: Coordinate = seeds_ds.voxel_size
    total_roi: Roi = affs_ds.roi

    write_roi = Roi(offset=(0,) * 3, shape=Coordinate(affs_ds.chunk_shape)[1:])

    min_neighborhood: int = min(
        filter(
            lambda x: x != 0, [value for sublist in neighborhood for value in sublist]
        )
    )
    max_neighborhood: int = max(
        filter(
            lambda x: x != 0, [value for sublist in neighborhood for value in sublist]
        )
    )

    read_roi: Roi = write_roi.grow(
        amount_neg=min_neighborhood, amount_pos=max_neighborhood
    )

    write_roi: Roi = write_roi * voxel_size
    read_roi: Roi = read_roi * voxel_size

    block_directory: str = os.path.join(fragments_file, "block_nodes")

    os.makedirs(name=block_directory, exist_ok=True)

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
        filename=fragments_file,
        ds_name=fragments_dataset,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=np.uint64,
        compressor={"id": "zlib", "level": 5},
        delete=True,
    )

    num_voxels_in_block = (write_roi / affs_ds.voxel_size).size

    # open RAG DB
    logging.info(msg="Opening RAG DB...")

    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    rag_provider = graphs.MongoDbGraphProvider(
        db_name=db_name,
        host=db_host,
        mode="w",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        edges_collection=f"hglom_edges_{merge_function}",
        nodes_collection=f"hglom_nodes",
        meta_collection=f"hglom_meta",
    )
    logging.info("RAG db opened")

    task: daisy.Task = daisy.Task(
        task_id="ExtractFragmentsTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda b: extract_fragments_worker(
            block=b,
            rag_provider=rag_provider,
            ds_in_file=affs_file,
            ds_in_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
            num_voxels_in_block=num_voxels_in_block,
            fragments_in_xy=fragments_in_xy,
            epsilon_agglomerate=epsilon_agglomerate,
            filter_fragments=filter_fragments,
            replace_sections=replace_sections,
            mask_file=mask_file,
            mask_dataset=mask_dataset,
        ),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("At least one block failed!")

    end: float = time.time()

    seconds: float = end - start
    minutes: float = seconds / 60
    hours: float = minutes / 60
    days: float = hours / 24

    print(
        "Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
    return done


def extract_fragments_worker(
    block,
    rag_provider,
    ds_in_file,
    ds_in_dataset,
    fragments_file,
    fragments_dataset,
    context,
    num_voxels_in_block,
    fragments_in_xy,
    epsilon_agglomerate,
    filter_fragments,
    replace_sections,
    mask_file,
    mask_dataset,
) -> None:
    logging.info("Reading ds_in from %s", ds_in_file)
    ds_in: Array = daisy.open_ds(filename=ds_in_file, ds_name=ds_in_dataset, mode="r")

    logging.info("Reading fragments from %s", fragments_file)
    fragments: Array = daisy.open_ds(
        filename=fragments_file, ds_name=fragments_dataset, mode="r+"
    )

    if mask_dataset is not None:
        logging.info(msg="Reading mask from {}".format(mask_file))
        mask: Array = daisy.open_ds(filename=mask_file, ds_name=mask_dataset, mode="r")

    else:
        mask = None

    logging.info("block read roi begin: %s", block.read_roi.offset)
    logging.info("block read roi shape: %s", block.read_roi.shape)
    logging.info("block write roi begin: %s", block.write_roi.offset)
    logging.info("block write roi shape: %s", block.write_roi.shape)

    watershed_in_block(
        affs=ds_in,
        block=block,
        context=context,
        rag_provider=rag_provider,
        fragments_out=fragments,
        num_voxels_in_block=num_voxels_in_block,
        mask=mask,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        filter_fragments=filter_fragments,
        replace_sections=replace_sections,
    )
