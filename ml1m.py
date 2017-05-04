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
	#ratings_df = pd.DataFrame()
	ratings_df = ratings_df.drop(["timestamp"], axis=1)
	ratings_df.to_csv(default_1m_rating_output_path, sep=" ",header=False, index=False)
	return ratings_df

def update_conf(conf, conf_opt, grid):
	print conf_opt
	for i in range(len(conf_opt)):
		conf.config[grid.keys()[i]] = grid[grid.keys()[i]][conf_opt[i]]

if __name__ == "__main__":
	SVD_grid = {
		"num.factors":[100,150,200],
		"reg.lambda":["-u %f -i %f -b %f -s %f"%(i,i,i,i) for i in [0, 0.001, 0.02]],
		"evaluation.setup":["-ap %f"%i for i in [0.9]]
	}
	svd_conf = config.Config("config/SVD.conf")
	grid = SVD_grid
	conf = svd_conf
	opts_arr = []
	for key in grid.keys():
		opts_arr.append(tuple(grid[key]))
	for opts in itertools.product(*opts_arr):
		for key, opt in zip(SVD_grid.keys(), opts):
			conf.config[key] = opt
		print svd_conf.config
		s = time.clock()
		recSys = RecQ(svd_conf)
		recSys.execute()
		e = time.clock()
		print "Run time: %f s" % (e - s)
		sys.stdout.flush()


