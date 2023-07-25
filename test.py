import heiarchical_agglom

pp: heiarchical_agglom.PostProcessor = heiarchical_agglom.PostProcessor(
            affs_file="./data/raw_predictions.zarr",
            affs_dataset="pred_affs_latest",
            seeds_file="./data/raw_predictions.zarr",
            seeds_dataset="training_gt_rasters",
        )
pp.run_hierarchical_agglom_segmentation_pipeline()
