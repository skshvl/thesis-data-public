


import pandas as pd



human_df = pd.read_pickle("../human_attribution_maps/3_combined_masks.pickle")


def get_human_map_for_filename(filename):
    """Get normalized and donwsampled 4x4 human attribution map for filename"""
    try:
        human_attribution_map = human_df[human_df['filename']==str(filename)]['mask_normalized_downsampled'].iloc[0]
        return human_attribution_map
    except Exception as e:
        print("Problem:", e)
        return None
    