#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import pandas as pd
import numpy as np
from tool import config
import time
from main.main import RecQ
import sys
import copy
import json
import itertools

default_1m_rating_path = "dataset/ml-1m/ratings.dat"
default_1m_rating_output_path = "dataset/ml-1m/ratings.csv"
default_1m_rating_shuffled_output_path = "dataset/ml-1m/ratings_%d.csv"
default_1m_rating_trainset_path = "dataset/ml-1m/ratings_trainset_%d_%.1f.csv"
default_1m_rating_testset_path = "dataset/ml-1m/ratings_testset_%d_%.1f.csv"

default_yelp_rating_path = "dataset/yelp/ratings.dat"
default_yelp_rating_output_path = "dataset/yelp/ratings.csv"
default_yelp_rating_shuffled_output_path = "dataset/yelp/ratings_%d.csv"
default_yelp_rating_trainset_path = "dataset/yelp/ratings_trainset_%d_%.1f.csv"
default_yelp_rating_testset_path = "dataset/yelp/ratings_testset_%d_%.1f.csv"

def build_df(objs, keys):
	df_dict = {}
	for key in keys:
		df_dict[key] = [obj[key] for obj in objs]
	df = pd.DataFrame(df_dict)
	return df

def shuffle(ratings_df, random_seed=0):
	return ratings_df.iloc[np.random.permutation(len(ratings_df))].reset_index(drop=True)

def split(ratings_df, trainset_rate=0.9):
	mid = int(ratings_df.shape[0]*trainset_rate)
	return ratings_df[:mid], ratings_df[mid:]

def yelp_preprocess():
	prefix = "dataset/yelp/"
	academic_dataset_json_prefix = "yelp_academic_dataset_%s.json"
	reviews = []

	for line in open(prefix+academic_dataset_json_prefix%("review")):
		reviews.append(json.loads(line))
	reviews_df = build_df(reviews, keys=['user_id', 'business_id', 'stars'])
	reviews_df = reviews_df[['user_id', 'business_id', 'stars']]
	reviews_df.to_csv(default_yelp_rating_path, sep=" ",header=False, index=False)
	seeds = [0, 1, 2, 3, 4]
	trainset_rates = [i * 0.1 for i in range(0, 10)]
	for seed in seeds:
		for rate in trainset_rates:
			shuffled_ratings_df = shuffle(copy.deepcopy(reviews_df), seed)
			shuffled_ratings_df.to_csv(default_yelp_rating_shuffled_output_path%(seed), sep=" ",header=False, index=False)

			trainset_df, testset_df = split(shuffled_ratings_df, rate)
			trainset_df.to_csv(default_yelp_rating_trainset_path%(seed, rate), sep=" ",header=False, index=False)
			testset_df.to_csv(default_yelp_rating_testset_path%(seed, rate), sep=" ",header=False, index=False)
	return reviews_df

def preprocess():
	ratings_df = pd.read_csv(default_1m_rating_path,sep="::", header=None,
	                      names=["user", "item", "rate", "timestamp"],
	                      engine="python")
	ratings_df = ratings_df.drop(["timestamp"], axis=1)
	ratings_df.to_csv(default_1m_rating_output_path, sep=" ",header=False, index=False)
	seeds = [0, 1, 2, 3, 4]
	trainset_rates = [i * 0.1 for i in range(0, 10)]
	for seed in seeds:
		for rate in trainset_rates:
			shuffled_ratings_df = shuffle(copy.deepcopy(ratings_df), seed)
			shuffled_ratings_df.to_csv(default_1m_rating_shuffled_output_path%(seed), sep=" ",header=False, index=False)

			trainset_df, testset_df = split(shuffled_ratings_df, rate)
			trainset_df.to_csv(default_1m_rating_trainset_path%(seed, rate), sep=" ",header=False, index=False)
			testset_df.to_csv(default_1m_rating_testset_path%(seed, rate), sep=" ",header=False, index=False)
	return ratings_df

def update_conf(conf, conf_opt, grid):
	print conf_opt
	for i in range(len(conf_opt)):
		conf.config[grid.keys()[i]] = grid[grid.keys()[i]][conf_opt[i]]


def grid_search_NMF(argv, data_type="ml1m"):
	from surprise import NMF, Dataset, evaluate, print_perf, Reader, GridSearch
	reader = Reader(line_format="user item rating", sep=' ')
	param_grid = {"n_epochs": [100], "n_factors": [10], "biased": [True]}
	grid_search = GridSearch(NMF, param_grid, verbose=2)
	seed_range = range(5)
	rate_range = [i*0.1 for i in [j for j in range(1,10)]]
	foldnames = gen_fold_names(seed_range, rate_range)
	print "FoldName:", foldnames
	if data_type == "ml1m":
		filenames = gen_fold_filenames(seed_range, rate_range)
	elif data_type == "yelp":
		filenames = gen_fold_filenames(seed_range, rate_range,
		                               trainset_path=default_yelp_rating_trainset_path,
		                               testset_path=default_yelp_rating_testset_path)
	data = Dataset.load_from_folds(filenames, reader)
	grid_search.evaluate(data)

	print 'grid_search.best_params["RMSE"]:', grid_search.best_params["RMSE"]
	print 'grid_search.best_score["RMSE"]:', grid_search.best_score["RMSE"]
	print 'grid_search.cv_results:', grid_search.cv_results

def gen_fold_names(seed_range=[0, 1, 2, 3, 4], rate_range=[0.1, 0.2, 0.3]):
	names = []
	for i in seed_range:
		for j in rate_range:
			names.append(str(i)+" "+str(j))
	return names

def gen_fold_filenames(seed_range=[0, 1, 2, 3, 4], rate_range=[0.1, 0.2, 0.3],
                       trainset_path=default_1m_rating_trainset_path, testset_path=default_1m_rating_testset_path):
	filenames = []
	for i in seed_range:
		for j in rate_range:
			filenames.append((trainset_path%(i, j), testset_path%(i, j)))
	return filenames

def run_conf(conf, grid={}):
	print "Run Conf %s"%conf.config['recommender']
	opts_arr = []
	for key in grid.keys():
		opts_arr.append(tuple(grid[key]))
	print [i for i in itertools.product(*opts_arr)]
	for opts in itertools.product(*opts_arr):
		for key, opt in zip(grid.keys(), opts):
			conf.config[key] = opt
		print conf.config
		s = time.clock()
		recSys = RecQ(conf)
		recSys.execute()
		e = time.clock()
		print "Run time: %f s" % (e - s)
		sys.stdout.flush()

if __name__ == "__main__":
	from sys import argv

	algo = argv[1]
	if algo == "svd":
		SVD_grid = {
			"num.factors":[200],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.02]],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		svd_conf = config.Config("config/SVD.conf")
		run_conf(svd_conf, SVD_grid)
	elif algo == "yelp_svd":
		SVD_grid = {
			"ratings":["dataset/yelp/ratings.csv"],
			"num.factors":[100],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.02]],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		svd_conf = config.Config("config/SVD.conf")
		run_conf(svd_conf, SVD_grid)
	elif algo == "svdpp":
		SVDPP_grid = {
			"num.factors":[150, 100 ,50],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.01, 0.005, 0.02]],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		svdpp_conf = config.Config("config/SVD++.conf")
		run_conf(svdpp_conf, SVDPP_grid)
	elif algo == "pmf":
		PMF_grid = {
			"num.factors":[150],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.02]],
			"evaluation.setup":["-ap %.1f"%i for i in [i * 0.1 for i in range(1, 10)]],
			"ratings": ["dataset/ml-1m/ratings_%d.csv"%i for i in [0, 1]]
		}
		pmf_conf = config.Config("config/PMF.conf")
		run_conf(pmf_conf,PMF_grid)
	elif algo == "yelp_pmf":
		PMF_grid = {
			"num.factors":[150, 50, 100],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.02, 0.005, 0.05]],
			"evaluation.setup":["-ap %.1f"%i for i in [i * 0.1 for i in range(9, 10)]],
			"ratings": ["dataset/yelp/ratings.csv"],
			"item.ranking": ["off -topN 0"],
			"output.setup": ["off -dir results/"]
		}
		pmf_conf = config.Config("config/PMF.conf")
		run_conf(pmf_conf, PMF_grid)
	elif algo == "slopeone":
		SlopeOne_grid = {
			"num.shrinkage":[30,20,40],
			"num.neighbors":[20,10,30],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		slope_one_conf = config.Config("config/SlopeOne.conf")
		run_conf(slope_one_conf, SlopeOne_grid)
	elif algo == "nmf":
		grid_search_NMF(argv)
	elif algo == "yelp_nmf":
		grid_search_NMF(argv, data_type="yelp")
	elif algo == "ml1m_preproc":
		preprocess()
	elif algo == "yelp_preproc":
		yelp_preprocess()
	else:
		raise Exception("Wrong algo name!")