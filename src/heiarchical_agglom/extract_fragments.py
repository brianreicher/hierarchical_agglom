import json
import logging
import numpy as np
import os
import daisy
import sys
import time
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, graphs
from lsd.post import watershed_in_block
from .utils import neighborhood

logging.getLogger().setLevel(logging.INFO)
#logging.getLogger('lsd.parallel_fragments').setLevel(logging.DEBUG)

def extract_fragments(
        base_dir,
        experiment,
        setup,
        iteration,
        file_name,
        ds_in_dataset,
        fragments_dataset,
        crops,
        block_size,
        context,
        num_workers,
        fragments_in_xy,
        epsilon_agglomerate=0,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0,
        replace_sections=None,
        **kwargs):
    
    '''Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.
    Args:
        ds_in_dataset,
        block_size (``tuple`` of ``int``):
            The size of one block in world units.
        context (``tuple`` of ``int``):
            The context to consider for fragment extraction and agglomeration,
            in world units.
        num_workers (``int``):
            How many blocks to run in parallel.
    '''

    for crop in crops:

        ds_in_file =  os.path.abspath(
                os.path.join(
                    base_dir,file_name
                    )
                )


        if crop != "":
            crop_path = os.path.join(mask_file,crop)
            
            with open(crop_path,"r") as f:
                crop = json.load(f)
            
            crop_name = crop["name"]
            crop_roi = daisy.Roi(crop["offset"],crop["shape"])

            ds_in_file = os.path.join(ds_in_file,crop_name+'.zarr')
            
        else:
            crop_name = ""
            crop_roi = None

        logging.info(f"Reading {ds_in_dataset} from {ds_in_file}")
        ds_in = daisy.open_ds(ds_in_file, ds_in_dataset, mode='r')

        if block_size == [0,0,0]: #if processing one block    
            context = [50,40,40]
            block_size = crop_roi.shape if crop_roi else ds_in.roi.shape
            
        raster_ds = open_ds(filename="../../../data/xpress-challenge.zarr", ds_name="volumes/training_gt_rasters")
        voxel_size = raster_ds.voxel_size
        dtype = raster_ds.dtype
        total_roi = ds_in.roi

        write_roi = daisy.Roi(offset=(0,)*3,shape=Coordinate(ds_in.chunk_shape)[1:])

        min_neighborhood: int = min( filter(lambda x: x != 0, [value for sublist in neighborhood for value in sublist]))
        max_neighborhood: int = max( filter(lambda x: x != 0, [value for sublist in neighborhood for value in sublist]))

        read_roi = write_roi.grow(amount_neg=min_neighborhood, amount_pos=max_neighborhood)

        write_roi = write_roi * voxel_size
        read_roi = read_roi * voxel_size

        fragments_file = ds_in_file

        block_directory = os.path.join(fragments_file,'block_nodes')

        os.makedirs(block_directory, exist_ok=True)

        # prepare fragments dataset
        fragments = daisy.prepare_ds(
            fragments_file,
            fragments_dataset,
            total_roi,
            voxel_size,
            np.uint64,
            compressor={'id': 'zlib', 'level':5},
            delete=True)

        num_voxels_in_block = (write_roi/ds_in.voxel_size).size

        task = daisy.Task(
            'ExtractFragmentsBlockwiseTask',
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: extract_fragments_worker(
                b,
                ds_in_file,
                ds_in_dataset,
                fragments_file,
                fragments_dataset,
                context,
                block_directory,
                write_roi.shape,
                num_voxels_in_block,
                fragments_in_xy,
                epsilon_agglomerate,
                filter_fragments,
                replace_sections,
                mask_file,
                mask_dataset),
            check_function=None,
            num_workers=num_workers,
            max_retries=7,
            read_write_conflict=False,
            fit='shrink')

        done = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("at least one block failed!")

        block_size = [0,0,0]

def extract_fragments_worker(
        block,
        ds_in_file,
        ds_in_dataset,
        fragments_file,
        fragments_dataset,
        context,
        block_directory,
        write_size,
        num_voxels_in_block,
        fragments_in_xy,
        epsilon_agglomerate,
        filter_fragments,
        replace_sections,
        mask_file,
        mask_dataset):

    logging.info("Reading ds_in from %s", ds_in_file)
    ds_in = daisy.open_ds(ds_in_file, ds_in_dataset, mode='r')

    logging.info("Reading fragments from %s", fragments_file)
    fragments = daisy.open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if mask_dataset is not None:

        logging.info("Reading mask from {}".format(mask_file))
        mask = daisy.open_ds(
            mask_file,
            mask_dataset,
            mode='r')

    else:

        mask = None

    # open RAG DB
    logging.info("Opening RAG DB...")

    
    db_host: str = "mongodb://localhost:27017"
    db_name: str = "seg"
    rag_provider = graphs.MongoDbGraphProvider(
        db_name=db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        edges_collection=f"hglom_edges",
        nodes_collection=f"hglom_nodes",
        meta_collection=f"hglom_meta",
    )
    logging.info("RAG file opened")

    logging.info("block read roi begin: %s", block.read_roi.offset)
    logging.info("block read roi shape: %s", block.read_roi.shape)
    logging.info("block write roi begin: %s", block.write_roi.offset)
    logging.info("block write roi shape: %s", block.write_roi.shape)

    watershed_in_block(
        ds_in,
        block,
        context,
        rag_provider,
        fragments,
        num_voxels_in_block=num_voxels_in_block,
        mask=mask,
        fragments_in_xy=fragments_in_xy,
        epsilon_agglomerate=epsilon_agglomerate,
        filter_fragments=filter_fragments,
        replace_sections=replace_sections)


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

        start = time.time()
        extract_fragments(**config)

        end = time.time()

        seconds = end - start
        minutes = seconds/60
        hours = minutes/60
        days = hours/24

        print('Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days' % (seconds, minutes, hours, days))