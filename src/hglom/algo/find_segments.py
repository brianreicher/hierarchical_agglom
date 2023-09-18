import logging
import multiprocessing as mp
import numpy as np
from funlib.geometry import Roi
from funlib.persistence import open_ds, graphs, Array
import os
import time
from funlib.segment.graphs.impl import connected_components

logging.getLogger().setLevel(logging.INFO)


def find_segments(
    affs_file: str,
    affs_dataset: str,
    fragments_file: str,
    fragments_dataset: str,
    thresholds_minmax: list = [0, 1],
    thresholds_step: float = 0.02,
    merge_function: str = "hist_quant_75",
) -> bool:
    """
    Args:
        fragments_file (``string``):
            Path to file (zarr/n5) containing fragments (supervoxels).
        edges_collection (``string``):
            The name of the MongoDB database edges collection to use.
        thresholds_minmax (``list`` of ``int``):
            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.
        thresholds_step (``float``):
            The step size to use when generating thresholds between min/max.
        block_size (``tuple`` of ``int``):
            The size of one block in world units (must be multiple of voxel
            size).
        num_workers (``int``):
            How many workers to use when reading the region adjacency graph
            blockwise.
        fragments_dataset (``string``, optional):
            Name of fragments dataset. Include if using full fragments roi, set
            to None if using a crop (roi_offset + roi_shape).
        run_type (``string``, optional):
            Can be used to direct luts into directory (e.g testing, validation,
            etc).
        roi_offset (array-like of ``int``, optional):
            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.
        roi_shape (array-like of ``int``, optional):
            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
    """

    logging.info("Reading graph")
    start: float = time.time()

    fragments: Array = open_ds(fragments_file, fragments_dataset)

    affs: Array = open_ds(filename=affs_file, ds_name=affs_dataset)
    roi: Roi = affs.roi

    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    graph_provider = graphs.MongoDbGraphProvider(
        db_name=db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        nodes_collection="hglom_nodes",
        edges_collection=f"hglom_edges_{merge_function}",
    )
    node_attrs: list = graph_provider.read_nodes(roi=roi)
    edge_attrs: list = graph_provider.read_edges(roi=roi, nodes=node_attrs)

    logging.info(msg=f"Read graph in {time.time() - start}")

    if "id" not in node_attrs[0]:
        logging.info(msg="No nodes found in roi %s" % roi)
        return

    nodes: list = [node["id"] for node in node_attrs]

    edge_u: list = [np.uint64(edge["u"]) for edge in edge_attrs]
    edge_v: list = [np.uint64(edge["v"]) for edge in edge_attrs]

    edges: np.ndarray = np.stack(arrays=[edge_u, edge_v], axis=1)

    scores: list = [np.float32(edge["merge_score"]) for edge in edge_attrs]

    logging.info(msg=f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir: str = os.path.join(fragments_file, "luts", "fragment_segment")

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [
        round(i, 2)
        for i in np.arange(
            float(thresholds_minmax[0]), float(thresholds_minmax[1]), thresholds_step
        )
    ]

    # parallel processing
    start = time.time()

    with mp.Pool(processes=4) as pool:
        pool.starmap(
            get_connected_components,
            [
                (
                    np.asarray(nodes),
                    np.asarray(edges),
                    np.asarray(scores),
                    t,
                    f"hglom_edges_{merge_function}",
                    out_dir,
                )
                for t in thresholds
            ],
        )

    logging.info(f"Created and stored lookup tables in {time.time() - start}")

    return True


def get_connected_components(
    nodes,
    edges,
    scores,
    threshold,
    edges_collection,
    out_dir,
):
    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*100)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)
