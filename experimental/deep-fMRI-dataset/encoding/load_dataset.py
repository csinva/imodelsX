"""
Get data from openneuro.
This will make a `data` directory in the main directory of the repo called data if it doesn't already exist.
"""
import os
import pathlib
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-download_preprocess", action="store_true",
						help="download data to disk. If true downloads all preprocessed data.")
	args = parser.parse_args()

	current_path = pathlib.Path(__file__).parent.resolve()
	main_dir = pathlib.Path(__file__).parent.parent.resolve()
	data_dir = os.path.join(main_dir, "data")

	os.chdir(main_dir)
	if not os.path.isdir(data_dir):               
		os.system("mkdir data")
		data_dir = os.path.join(main_dir, "data")
	os.chdir(data_dir)
	os.system("datalad clone https://github.com/OpenNeuroDatasets/ds003020")
	if args.download_preprocess == True:
		os.chdir("ds003020")
		os.system("datalad get derivative")
