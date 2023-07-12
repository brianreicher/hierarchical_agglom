from .extract_fragments import extract_fragments


def run_hierarchical_agglom_segmentation(affs_file,
                                         affs_dataset,
                                         fragments_file,
                                         fragments_dataset,
                                         seeds_file,
                                         seeds_dataset,
                                         context) -> bool:
    
    _: bool = extract_fragments(affs_file=affs_file,
                                affs_dataset=affs_dataset,
                                fragments_file=fragments_file
                                fragments_dataset=fragments_dataset,
                                seeds_file=seeds_file,
                                seeds_dataset=seeds_dataset,
                                context=context,)

    return True