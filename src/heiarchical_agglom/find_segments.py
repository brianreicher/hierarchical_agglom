import daisy
import json
import logging
import multiprocessing as mp
import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, Array, graphs, prepare_ds
import os
import sys
import time
from funlib.segment.graphs.impl import connected_components

logging.getLogger().setLevel(logging.INFO)

def find_segments(
        fragments_file,
        edges_collection,
        thresholds_minmax,
        thresholds_step,
        block_size,
        num_workers,
        crops,
        fragments_dataset=None,
        run_type=None,
        roi_offset=None,
        roi_shape=None,
        **kwargs):

    '''
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
    '''


    for crop in crops:
    
        logging.info("Reading graph")
        start = time.time()
        
        if crop != "":
            fragments_file = os.path.join(fragments_file,os.path.basename(crop)[:-4]+'zarr')
            crop_path = os.path.join(fragments_file,'crop.json')
            with open(crop_path,"r") as f:
                crop = json.load(f)
            
            crop_name = crop["name"]
            crop_roi = daisy.Roi(crop["offset"],crop["shape"])

        else:
            crop_name = ""
            crop_roi = None

        print(f"FRAGS FILE {fragments_file}")
        # block_directory = os.path.join(fragments_file,'block_nodes')

        fragments = open_ds(fragments_file,fragments_dataset)

        # if block_size == [0,0,0]: #if processing one block    
        #     context = [50,40,40]
        #     block_size = crop_roi.shape if crop_roi else fragments.roi.shape
        affs = open_ds(fragments_file, "pred_affs_latest")
        roi = affs.roi

        # block_size = Coordinate(block_size)

        db_host: str = "mongodb://localhost:27017"
        db_name: str = "seg"
        graph_provider = graphs.MongoDbGraphProvider(
            db_name=db_name,
            host=db_host,
            mode="r+",
            directed=False,
            position_attribute=["center_z", "center_y", "center_x"],
            nodes_collection="hglom_nodes",
            edges_collection=edges_collection,
        )
        node_attrs = graph_provider.read_nodes(roi)
        edge_attrs = graph_provider.read_edges(roi)
        edge_attrs = graph_provider.read_edges(roi,nodes=node_attrs)

        # node_attrs,edge_attrs = rag_provider.read_blockwise(roi,block_size/2,num_workers)

        logging.info(f"Read graph in {time.time() - start}")

        if 'id' not in node_attrs[0]:
            logging.info('No nodes found in roi %s' % roi)
            return

        nodes: list = [node['id'] for node in node_attrs]
        
        edge_u: list = [np.uint64(edge['u']) for edge in edge_attrs]
        edge_v: list = [np.uint64(edge['v']) for edge in edge_attrs]

        edges: np.ndarray = np.stack(
                    arrays=[
                        edge_u,
                        edge_v
                    ],
                axis=1)

        scores: list = [np.float32(edge["merge_score"]) for edge in edge_attrs]

        # scores = edge_attrs['merge_score'].astype(np.float32)

        logging.info(f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

        out_dir = os.path.join(
            fragments_file,
            'luts',
            'fragment_segment')

        if run_type:
            out_dir = os.path.join(out_dir, run_type)

        os.makedirs(out_dir, exist_ok=True)

        thresholds = [round(i,2) for i in np.arange(
            float(thresholds_minmax[0]),
            float(thresholds_minmax[1]),
            thresholds_step)]

        #parallel processing
        
        start = time.time()

        with mp.Pool(4) as pool:

            pool.starmap(get_connected_components,[(np.asarray(nodes),np.asarray(edges),np.asarray(scores),t,edges_collection,out_dir) for t in thresholds])

        logging.info(f"Created and stored lookup tables in {time.time() - start}")

        #reset
        block_size = [0,0,0]
        fragments_file = os.path.dirname(fragments_file)

def get_connected_components(
        nodes,
        edges,
        scores,
        threshold,
        edges_collection,
        out_dir,
        **kwargs):

    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*100)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    start = time.time()
    find_segments(**config)

    logging.info(f'Took {time.time() - start} seconds to find segments and store LUTs')