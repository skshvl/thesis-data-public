# from ALBEF

import numpy as np
import shap
import torch
import pandas as pd
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os, copy, json
import re, math, sys
import random
from tqdm import tqdm
from functools import partial

# from ALBEF.models.vit import VisionTransformer
# from ALBEF.models.xbert import BertConfig, BertModel
# from ALBEF.models.tokenization_bert import BertTokenizer

# from CLIP

from transformers import CLIPProcessor, CLIPModel


use_cuda = True # for ALBEF

ALBEF_PATH = 'ALBEF/checkpoints/ALBEF_4M.pth'


class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert=''
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


def load_models(which="clip"):
    """ Load models and model components. """

    if which == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    
    if which == "albef":
        model_path = ALBEF_PATH  # largest model: ALBEF.pth, smaller: ALBEF_4M.pth, refcoco, mscoco, vqa, flickr30k
        bert_config_path = 'ALBEF/configs/config_bert.json'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        model = VL_Transformer_ITM(
            text_encoder='bert-base-uncased', config_bert=bert_config_path)

        checkpoint = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        model.eval()

        block_num = 8

        model.text_encoder.base_model.base_model.encoder.layer[
            block_num].crossattention.self.save_attention = True

        if use_cuda:
            model.cuda()
        return model, tokenizer

def compute_mm_score(text_length, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    text_contrib = np.abs(shap_values.values[0, 0, :text_length]).sum()
    image_contrib = np.abs(shap_values.values[0, 0, text_length:]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score


VALSE_DATA_PATH = "../VALSE_data/valse_challenges_with_img_paths.csv"

def load_valse_data(n_samples = 2, filename_caption_foil_trios = None, ling_phenomena=None):
    """Return: List of dictionaries with elements "img_path", "caption", "foil", "linguistic_phenomena"
    if n_samples = "all", loads ALL
    ling_phenomena should be a LIST
    filenames should be a LIST

    if filenames given, ling_phenomena is IGNORED
    """

    df = pd.read_csv(VALSE_DATA_PATH, converters={"mturk": eval}) # evaluate mturk column as actual dictionaries

    # only accept images accepted by MTURK annotators (need to load string version of each row's dict into dict thru json.loads)
    df = df[df['mturk'].apply(lambda x: x['caption'] >= 2)]

    # if given list of filenames, filter by those:
    if filename_caption_foil_trios:
        mask = df.apply(lambda row: (row['image_file'], row['caption'], row['foil']) in filename_caption_foil_trios, axis=1)
        df = df[mask]
        print(f"Filtered VALSE data to only given filenames/caption/foil trios ({len(filename_caption_foil_trios)}), down to {len(df)} images. Among them {len(set(list(df['image_file'])))} unique filenames")

        df = df.drop_duplicates(subset=['image_file', 'caption', 'foil'])
        print("Removed any duplicates of image + caption + foil. Now we have {len(df)} total stimuli")
   
    # filter by ling phenomena ONLY IF NOT FILTERING BY FILENAME
    if ling_phenomena:
        df = df[df["linguistic_phenomena"].isin(ling_phenomena)]
        print(f"Filtered VALSE data to only {ling_phenomena}, down to {len(df)} images.")

    # sample n_samples rows
    if n_samples != "all": 
        df = df.sample(n_samples, random_state = 42)

    output_data = []

    # output the data as a LIST of image paths, captions, foils
    for index, row in df.iterrows():
        output_data.append(
            {
                # add image path (add VALSE_data prefix because image paths are relative to VALSE_data folder)
                "img_path":os.path.join("../VALSE_data", row["local_img_path"]),

                # caption and foil
                "caption":row["caption"],
                "foil":row["foil"],

                "linguistic_phenomena":row["linguistic_phenomena"]
            }
        )

    print(f"Data loaded with sampling: {len(output_data)} rows")
    
    return output_data




