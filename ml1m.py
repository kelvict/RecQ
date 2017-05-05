#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import pandas as pd
from tool import config
import time
from main.main import RecQ
import sys
import copy

import itertools
def preprocess():
	default_1m_rating_path = "dataset/ml-1m/ratings.dat"
	default_1m_rating_output_path = "dataset/ml-1m/ratings.csv"
	ratings_df = pd.read_csv(default_1m_rating_path,sep="::", header=None,
	                      names=["user", "item", "rate", "timestamp"],
	                      engine="python")
	#claratings_df = pd.DataFrame()
	ratings_df = ratings_df.drop(["timestamp"], axis=1)
	ratings_df.to_csv(default_1m_rating_output_path, sep=" ",header=False, index=False)
	return ratings_df

def update_conf(conf, conf_opt, grid):
	print conf_opt
	for i in range(len(conf_opt)):
		conf.config[grid.keys()[i]] = grid[grid.keys()[i]][conf_opt[i]]

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
			"num.factors":[50,100,150],
			"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0.01, 0.005, 0.02]],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		pmf_conf = config.Config("config/PMF.conf")
		run_conf(pmf_conf,PMF_grid)
	elif algo == "slopeone":
		SlopeOne_grid = {
			"num.shrinkage":[30,20,40],
			"num.neighbors":[20,10,30],
			"evaluation.setup":["-ap %f"%i for i in [0.1]]
		}
		slope_one_conf = config.Config("config/SlopeOne.conf")
		run_conf(slope_one_conf, SlopeOne_grid)
	else:
		raise Exception("Wrong algo name!")
