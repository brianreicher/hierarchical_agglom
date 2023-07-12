import daisy
import json
import logging
import numpy as np
import os
import sys
import time
from funlib.segment.arrays import replace_values


logging.getLogger().setLevel(logging.INFO)

def extract_segmentation(
        fragments_file,
        fragments_dataset,
        crops,
        edges_collection,
        block_size,
        threshold,
        out_dataset,
        num_workers,
        roi_offset=None,
        roi_shape=None,
        run_type=None,
        **kwargs):

    '''
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
    '''

    for crop in crops:
        # open fragments

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
        
        results_file = os.path.join(fragments_file,"results.json")
        
        lut_dir = os.path.join(fragments_file,'luts','fragment_segment')

        fragments = daisy.open_ds(fragments_file, fragments_dataset)

        if block_size == [0,0,0]:
            context = [50,40,40]
            block_size = crop_roi.shape if crop_roi else fragments.roi.shape

        total_roi = fragments.roi
        read_roi = daisy.Roi((0,)*3, daisy.Coordinate(block_size))
        write_roi = read_roi

        logging.info("Preparing segmentation dataset...")

        thresholds = [.64, .74, .84, .94]
        
        if os.path.exists(results_file):
            with open(results_file,"r") as f:
                results = json.load(f)
                bests = [results[x]['best_voi']['threshold'] for x in results.keys()]
                for best in bests:
                    if best not in thresholds:
                        thresholds.append(best)

        for threshold in thresholds:

            seg_name = f"segmentation_{threshold}"

            start = time.time()

            segmentation = daisy.prepare_ds(
                fragments_file,
                seg_name,
                fragments.roi,
                voxel_size=fragments.voxel_size,
                dtype=np.uint64,
                write_roi=write_roi,
                delete=True)

            lut_filename = f'seg_{edges_collection}_{int(threshold*100)}'

            lut = os.path.join(
                    lut_dir,
                    lut_filename + '.npz')

            assert os.path.exists(lut), f"{lut} does not exist"

            logging.info("Reading fragment-segment LUT...")

            lut = np.load(lut)['fragment_segment_lut']

            logging.info(f"Found {len(lut[0])} fragments in LUT")

            num_segments = len(np.unique(lut[1]))
            logging.info(f"Relabelling fragments to {num_segments} segments")

            task = daisy.Task(
                'ExtractSegmentationTask',
                total_roi,
                read_roi,
                write_roi,
                lambda b: segment_in_block(
                    b,
                    segmentation,
                    fragments,
                    lut),
                fit='shrink',
                num_workers=num_workers)

            done = daisy.run_blockwise([task])

            if not done:
                raise RuntimeError("Extraction of segmentation from LUT failed for (at least) one block")

            logging.info(f"Took {time.time() - start} seconds to extract segmentation from LUT")
   
        #reset
        block_size = [0,0,0]
        fragments_file = os.path.dirname(fragments_file)

def segment_in_block(
        block,
        segmentation,
        fragments,
        lut):

    logging.info("Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.write_roi)

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    relabelled = replace_values(fragments, lut[0], lut[1], out_array=relabelled)

    segmentation[block.write_roi] = relabelled

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    print("extracting segmentation . . .")
    extract_segmentation(**config)