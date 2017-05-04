#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import pandas as pd
from tool import config
import time
from main.main import RecQ
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

if __name__ == "__main__":
	conf = config.Config("config/test.conf")
	print conf
	s = time.clock()
	recSys = RecQ(conf)
	recSys.execute()
	e = time.clock()
	print "Run time: %f s" % (e - s)