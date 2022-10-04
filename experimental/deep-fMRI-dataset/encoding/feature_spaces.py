import os
import sys
import numpy as np
import json
from os.path import join, dirname

from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.dsutils import make_semantic_model, make_word_ds, make_phoneme_ds
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles

repo_dir = join(dirname(dirname(os.path.abspath(__file__))))
em_data_dir = join(repo_dir, 'em_data')
data_dir = join(repo_dir, 'encoding', 'data')

def get_story_wordseqs(stories):
	grids = load_textgrids(stories, data_dir)
	with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs

def get_story_phonseqs(stories):
	grids = load_textgrids(stories, data_dir)
	with open(join(data_dir, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_phoneme_ds(grids, trfiles)
	return wordseqs

def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

###########################################
########## ARTICULATORY Features ##########
###########################################

def ph_to_articulate(ds, ph_2_art):
	""" Following make_phoneme_ds converts the phoneme DataSequence object to an 
	articulate Datasequence for each grid.
	"""
	articulate_ds = []
	for ph in ds:
		try:
			articulate_ds.append(ph_2_art[ph])
		except:
			articulate_ds.append([""])
	return articulate_ds

articulates = ["bilabial","postalveolar","alveolar","dental","labiodental",
			   "velar","glottal","palatal", "plosive","affricative","fricative",
			   "nasal","lateral","approximant","voiced","unvoiced","low", "mid",
			   "high","front","central","back"]

def histogram_articulates(ds, data, articulateset=articulates):
	"""Histograms the articulates in the DataSequence [ds]."""
	final_data = []
	for art in ds:
		final_data.append(np.isin(articulateset, art))
	final_data = np.array(final_data)
	return (final_data, data.split_inds, data.data_times, data.tr_times)

def get_articulation_vectors(allstories):
	"""Get downsampled articulation vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(em_data_dir, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			arthistseq[0], arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs


###########################################
########## PHONEME RATE Features ##########
###########################################

def get_phonemerate_vectors(allstories):
	"""Get downsampled phonemerate vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(em_data_dir, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		nphonemes = arthistseq[0].shape[0]
		phonemerate = np.ones([nphonemes, 1])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			phonemerate, arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs

########################################
########## WORD RATE Features ##########
########################################

def get_wordrate_vectors(allstories):
	"""Get wordrate vectors for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		nwords = len(wordseqs[story].data)
		vectors[story] = np.ones([nwords, 1])
	return downsample_word_vectors(allstories, vectors, wordseqs)


######################################
########## ENG1000 Features ##########
######################################

def get_eng1000_vectors(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(em_data_dir, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
	return downsample_word_vectors(allstories, vectors, wordseqs)

############################################
########## Feature Space Creation ##########
############################################

_FEATURE_CONFIG = {
	"articulation": get_articulation_vectors,
	"phonemerate": get_phonemerate_vectors,
	"wordrate": get_wordrate_vectors,
	"eng1000": get_eng1000_vectors,
}

def get_feature_space(feature, *args):
	return _FEATURE_CONFIG[feature](*args)
