from .extract_fragments import extract_fragments
from .agglomerate_blockwise import agglomerate
from .find_segments import find_segments
from .extract_segments_from_lut import extract_segmentation


def run_hierarchical_agglom_segmentation_pipeline(
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    seeds_file,
    seeds_dataset,
    context,
) -> bool:
    success: bool = extract_fragments(
        affs_file=affs_file,
        affs_dataset=affs_dataset,
        fragments_file=fragments_file,
        fragments_dataset=fragments_dataset,
        seeds_file=seeds_file,
        seeds_dataset=seeds_dataset,
        context=context,
    )
    if success:
        success: bool = agglomerate(
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
            context=context,
        )

    if success:
        success: bool = find_segments(
            affs_file=affs_file,
            affs_dataset=affs_dataset,
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
        )

    if success:
        success: bool = extract_segmentation(
            fragments_file=fragments_file,
            fragments_dataset=fragments_dataset,
        )
    return success
