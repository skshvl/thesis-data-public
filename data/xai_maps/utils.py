import pandas as pd
import ast

MACHINE = "offline"


if MACHINE == "colab":
    stim_paths = [ f"stimuli/stimuli_{i}_fixed_clip_values.pickle" for i in [1,2,3]]
    STIMULI_DF = pd.concat([pd.read_pickle(path) for path in stim_paths]).reset_index() 

else:
    # concatenate all stimuli into a single dataframe
    stim_paths = [ f"../../data_prep/final_stimuli/stimuli_{i}_fixed_clip_values.pickle" for i in [1,2,3]]
    STIMULI_DF = pd.concat([pd.read_pickle(path) for path in stim_paths]).reset_index()

    ENTIRE_DATASET_DF = pd.read_csv("../../VALSE_data/valse_challenges_with_img_paths.csv")

    # now exlude non-validated VALSE stimuli from the whole dataset -- based on having at least 2/3 mturk respondents from original VALSE study clicking "caption"
    include_in_experiment = []
    for mturk_dict_str in ENTIRE_DATASET_DF['mturk']:
        mturk_dict = ast.literal_eval(mturk_dict_str)
        if dict(mturk_dict)['caption'] >= 2:
            include_in_experiment.append(True)
        else:
            include_in_experiment.append(False)
    ENTIRE_DATASET_DF['include_in_experiment'] = include_in_experiment

    # filter to only validated examples
    ENTIRE_DATASET_DF = ENTIRE_DATASET_DF[ENTIRE_DATASET_DF['include_in_experiment']==True]
