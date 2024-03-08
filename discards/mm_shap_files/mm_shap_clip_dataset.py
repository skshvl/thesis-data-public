# conda activate shap (rampage)
import shap
import torch
import numpy as np
from PIL import Image
import os, copy, sys
import math, json
import random
from tqdm import tqdm

import pickle
import json
import pandas as pd
import os

from collections import defaultdict

# Get the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the current script's directory
os.chdir(current_directory)

from shap_utils import load_valse_data, compute_mm_score, load_models

# Original version of this code can be found at: https://github.com/Heidelberg-NLP/MM-SHAP/blob/main/mm-shap_clip_dataset.py

# POSSIBLE LINGUISTIC PHENOMENA:
# {'actions',
#  'coreference',
#  'counting',
#  'existence',
#  'noun phrases',
#  'plurals',
#  'relations'}


def custom_masker_bimodal(mask, x):
    """
    Shap relevant function. Defines the masking function so the shap computation
    can 'know' how the model prediction looks like when some tokens are masked.
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    #
    #  never mask out CLS and SEP tokens (makes no sense for the model to work without them)
    masked_X[0, 0] = 49406
    masked_X[0, text_length_tok-1] = 49407

    # print(f"Masking X in bimodal SHAP with text length {text_length_tok}. Masked X:", masked_X)
    
    return masked_X

def custom_masker_image_only_no_text_output(mask, x):
    """
    Shap relevant function. Defines the masking function so the shap computation
    can 'know' how the model prediction looks like when some tokens are masked.

    mask is only the length of IMAGE!

    NO TEXT TOKENS IN OUTPUT
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero

    # add UNMASKED TEXT tokens
    masked_X = torch.tensor(masked_X)

    #print(f"Masking image in SHAP. Masked X:", masked_X)
    
    return masked_X

def get_clip_prediction_two_captions(x_stacked):
    """x is image masking map, get difference of two caption predictions (captions taken from inputs variable)"""
    
    with torch.no_grad():
        row_cols = 224 // patch_size # 224 / 32 = 7
        result = np.zeros((x_stacked.shape[0]))
        masked_image_token_ids = torch.tensor(x_stacked)
        

        for row in range(x_stacked.shape[0]):
            masked_inputs = copy.deepcopy(inputs)
            for token_position in range(masked_image_token_ids[row].shape[0]):
                if masked_image_token_ids[row][token_position] == 0:  # should be zero
                    m = token_position // row_cols
                    n = token_position % row_cols
                    masked_inputs["pixel_values"][:, :, m *
                        patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()
            
                # 
            outputs = torch.tensor(clip_model(**masked_inputs).logits_per_text)

            # based on global variable CLIP_RETURN, we can choose whether it returns the difference, caption or foil score
            if CLIP_RETURN == "diff":
                result[row] = outputs[0,0]-outputs[1,0]
            elif CLIP_RETURN == "caption":
                result[row] = outputs[0,0]
            elif CLIP_RETURN == "foil":
                result[row] = outputs[1,0]
            else:
                print("NO RETURN INSTRUCTION TO CLIP. ABORTING")
                sys.exit()
            # print(f"Mask: {masked_image_token_ids[row]}\n Outputs: {outputs}, difference: {result[row]}")
    
    return result

def get_clip_prediction_one_caption(x):
    """
    Shap relevant function. Predict the model output for all combinations of masked tokens.
    EDUARD: This does the following:
    - Copy the masked input mapping (paramter x) into text and image parts (originally it's a concatenation)
    - Use the image of x to part to generate a new image based on the original input variable (global)
    - Copy x's masked text (input_ids) and the masked image both into masked_inputs variable
    - Use the masked_inputs variable (consisting of input_ids and pixel_values, both masked) to make a model prediction and get the logit result
    - Theoretically it works for multiple captions at once, but in practice we don't do it that way.
    - All this is with x being the OUTPUT of a shapley being done to the X input
    """
    with torch.no_grad():
        input_ids = torch.tensor(x[:, :inputs.input_ids.shape[1]])
        masked_image_token_ids = torch.tensor(x[:, inputs.input_ids.shape[1]:])
        result = np.zeros(input_ids.shape[0])
        row_cols = 224 // patch_size # 224 / 32 = 7
        for i in range(input_ids.shape[0]): # loop through x instances, which can be MULTIPLE

            masked_inputs = copy.deepcopy(inputs)  # initialize the masked inputs variable which will be modified version of current inputs
            masked_inputs['input_ids'] = input_ids[i].unsqueeze(0) # this is the text part of the x data (already masked)
            for k in range(masked_image_token_ids[i].shape[0]): # loop through token indexes
                if masked_image_token_ids[i][k] == 0:  # if zero, mask region corresponding to token k
                    m = k // row_cols
                    n = k % row_cols
                    masked_inputs["pixel_values"][:, :, m *
                        patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()
            
            # 
            outputs = clip_model(**masked_inputs)
            result[i] = outputs.logits_per_image

    return result



clip_model, clip_processor = load_models("clip")

# load VALSE data

# get stimuli table and generate list of filenames/captions
STIMULI_PATH = "../data_prep_and_analysis/jan8_preliminary_sampled_stimuli.pickle"
stimuli_df = pd.read_pickle(STIMULI_PATH)
# make list of tuples: image filename, caption
filename_caption_pairs = [(os.path.basename(img_path), caption) for img_path, caption in zip(stimuli_df['img_path'], stimuli_df['caption'])]

# load VALSE data for only those filenames
data_list = load_valse_data(n_samples = "all", filename_caption_pairs = filename_caption_pairs,
                            ling_phenomena = ["existence", "relations", "actions"]) # ling phenomena IGNORED if filenames given

MODEL = "clip"
UNIMODAL_SHAP = False # whether to do unimodal shap, only masking images
MULTIMODAL_SHAP = False # whether to calculate multimodality scores and use bimodal masking to do it

# we output results of MMSHAP evaluation into a dictionary first, then into a dataframe
results = defaultdict(list)


for data_point in tqdm(data_list):

    # copy over those things that should be copied directly
    for name in ["img_path", "caption", "foil", "linguistic_phenomena"]:
        results[name].append(data_point[name])

    # prepare data point as inputs to test model on
    image = Image.open(data_point["img_path"])

    test_sentences = [data_point["caption"],
                      data_point["foil"]]
    
    # create CLIP input with BOTH captions
    try:
        inputs = clip_processor(
            text = test_sentences,
            images = image,
            return_tensors = 'pt',
            padding = True
        )

    except:
        print(f"Trouble processing {data_point['img_path']}")
        continue
    
    clip_logits = clip_model(**inputs).logits_per_image[0,0:2].tolist()
    clip_cap_foil_diff = clip_logits[0] - clip_logits[1]

    # save relevant fields to the dictionary to either {caption} or {foil} (based on which it is)
    results["clip_pred_caption"].append(clip_logits[0]) # logit
    results["clip_pred_foil"].append(clip_logits[1])
    results["clip_pred_diff"].append(clip_cap_foil_diff)

    if UNIMODAL_SHAP:
        # Set up SHAP
        NR_ROWS, patch_size = 4, 224 // 4 # patch size should divide by same number as NR_ROWS
        image_token_ids = torch.tensor( range(1, NR_ROWS**2+1)).unsqueeze(0) #p x p patches        
        X = image_token_ids.unsqueeze(1) # only image token ids are passed to the explainer

        clip_explainer = shap.Explainer(
                get_clip_prediction_two_captions, custom_masker_image_only_no_text_output, silent=False)
        
        # do SHAP explanations with different values of CLIP return
        # This variable determines if clip output is score for caption, foil, or difference
        CLIP_RETURN = "diff"
        clip_shap_diff = clip_explainer(X)
        CLIP_RETURN = "caption"
        clip_shap_caption = clip_explainer(X)
        CLIP_RETURN = "foil"
        clip_shap_foil = clip_explainer(X)


        results[f"clip_shap_diff"].append(clip_shap_diff.values)
        results[f"clip_shap_caption"].append(clip_shap_caption.values)
        results[f"clip_shap_foil"].append(clip_shap_foil.values)

    # MULTIMODALITY calculation - pass each caption-img pair separately and run bimodal SHAP
    if MULTIMODAL_SHAP:
        # create new inputs variable for this sentence-img pair
        for k, sentence in enumerate(test_sentences):
            try:  # image feature extraction can go wrong
                inputs = clip_processor(
                    text=sentence, images=image, return_tensors="pt", padding=True
                )
            except:
                print(f"Trouble processing {data_point['img_path']}")
                continue

            text_length_tok = inputs.input_ids.shape[1] # nr of text tokens

            p = int(math.ceil(np.sqrt(text_length_tok)))
            patch_size = 224 // p # determine patch nr and patch size based on nr of tokens assuming 224x224 image
            image_token_ids = torch.tensor( 
                range(1, p**2+1)).unsqueeze(0) #p x p patches        
            # NOTE: There is no image data in X;The actual image pixel data is accessed by get_clip_prediction() from the inputs variable
            X = torch.cat(
                (inputs.input_ids, image_token_ids), 1).unsqueeze(1)

            # create an explainer with model and image masker
            explainer2 = shap.Explainer(
                get_clip_prediction_one_caption, custom_masker_bimodal, silent=True)
            shap_values = explainer2(X)

            mm_score = compute_mm_score(text_length_tok, shap_values)

            # if it's CAPTION (so k=0), update the data for caption        
            if k == 0:
                which = 'caption'
            # if it's FOIL (so k=1), update the data for foil
            else:
                which = 'foil'

            # save relevant fields to the dictionary to either {caption} or {foil} (based on which it is)
            results[f"mm_score_"+which].append(mm_score) #multimodality score
            results[f"bimodal_shap_values_"+which].append(shap_values.values) # shap values array
            results[f"text_tok_"+which].append(text_length_tok) # number of text tokens

results_df = pd.DataFrame(results)

results_df.to_pickle("../data_analysis/clip_all_results.pickle")