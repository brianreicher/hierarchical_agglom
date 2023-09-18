import logging
import daisy
from funlib.geometry import Coordinate, Roi
from funlib.persistence import graphs, Array
import time
from ..utils import neighborhood
from lsd.post import agglomerate_in_block

logging.getLogger().setLevel(logging.INFO)


def agglomerate(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    context: tuple,
    num_workers: int = 7,
    merge_function: str = "hist_quant_75",
) -> None:
    """Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.
    Args:
        file_name (``string``):
            The input file containing affs and fragments.
        affs_dataset, fragments_dataset (``string``):
            Where to find the affinities and fragments.
        block_size (``tuple`` of ``int``):
            The size of one block in world units.
        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.
        num_workers (``int``):
            How many blocks to run in parallel.
        merge_function (``string``):
            Symbolic name of a merge function. See dictionary below.
    """

    start: float = time.time()
    logging.info(msg=f"Reading {affs_dataset} from {affs_file}")

    fragments: Array = daisy.open_ds(
        filename=fragments_file, ds_name=fragments_dataset, mode="r"
    )

    voxel_size: Coordinate = fragments.voxel_size
    total_roi: Roi = fragments.roi

    write_roi = daisy.Roi(offset=(0,) * 3, shape=Coordinate(fragments.chunk_shape))

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

    read_roi: Roi = (
        write_roi  # .grow(amount_neg=min_neighborhood, amount_pos=max_neighborhood)
    )

    write_roi: Roi = write_roi * voxel_size
    read_roi: Roi = read_roi * voxel_size

    logging.info("Reading fragments from %s", fragments_file)

    context = Coordinate(context)

    total_roi = fragments.roi
    waterz_merge_function: dict = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs: Array = daisy.open_ds(filename=affs_file, ds_name=affs_dataset)

    # opening RAG file
    logging.info(msg="Opening RAG file...")

    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    rag_provider = graphs.MongoDbGraphProvider(
        db_name=db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        nodes_collection="hglom_nodes",
        edges_collection=f"hglom_edges_{merge_function}",
    )

    logging.info(msg="RAG file opened")

    task = daisy.Task(
        task_id="AgglomerateTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda block: agglomerate_in_block(
            affs=affs,
            fragments=fragments,
            rag_provider=rag_provider,
            block=block,
            merge_function=waterz_merge_function,
            threshold=1.0,
        ),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("at least one block failed!")

    end: float = time.time()

    seconds: float = end - start
    minutes: float = seconds / 60
    hours: float = minutes / 60
    days: float = hours / 24

    print(
        "Total time to agglomerate fragments: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
    return done


def agglomerate_worker(
    block,
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    merge_function: str = "hist_quant_75",
) -> None:
    waterz_merge_function: dict = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs: Array = daisy.open_ds(filename=affs_file, ds_name=affs_dataset)

    logging.info(f"Reading fragments from {fragments_file}")
    fragments: Array = daisy.open_ds(filename=fragments_file, ds_name=fragments_dataset)

    # opening RAG file
    logging.info(msg="Opening RAG file...")

    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    rag_provider = graphs.MongoDbGraphProvider(
        db_name=db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        nodes_collection="hglom_nodes",
        edges_collection=f"hglom_edges_{merge_function}",
    )
    logging.info(msg="RAG file opened")

    agglomerate_in_block(
        affs=affs,
        fragments=fragments,
        rag_provider=rag_provider,
        block=block,
        merge_function=waterz_merge_function,
        threshold=1.0,
    )
