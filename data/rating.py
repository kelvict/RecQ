import numpy as np
from structure import sparseMatrix,new_sparseMatrix
from tool.config import Config,LineConfig
from tool.qmath import normalize
from evaluation.dataSplit import DataSplit
import os.path
from re import split

class RatingDAO(object):
    'data access control'
    def __init__(self,config,trainingSet = list(), testSet = list()):
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.user = {} #used to store the order of users in the training set
        self.item = {} #used to store the order of items in the training set
        self.id2user = {}
        self.id2item = {}
        self.all_Item = {}
        self.all_User = {}
        self.userMeans = {} #used to store the mean values of users's ratings
        self.itemMeans = {} #used to store the mean values of items's ratings
        self.trainingData = [] #training data
        self.testData = [] #testData
        self.globalMean = 0
        self.timestamp = {}
        self.trainingMatrix = None
        self.validationMatrix = None
        self.testSet_u = {} # used to store the test set by hierarchy user:[item,rating]
        self.testSet_i = {} # used to store the test set by hierarchy item:[user,rating]
        self.rScale = []

        self.trainingData = trainingSet
        self.testData = testSet
        self.__generateSet()

        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()



    def __generateSet(self):
        triple = []
        scale = set()
        # find the maximum rating and minimum value
        for i, entry in enumerate(self.trainingData):
            userName, itemName, rating = entry
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()

        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # makes the rating within the range [0, 1].
            rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            self.trainingData[i][2] = rating
            # order the user
            if not self.user.has_key(userName):
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if not self.item.has_key(itemName):
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            triple.append([self.user[userName], self.item[itemName], rating])
        self.trainingMatrix = new_sparseMatrix.SparseMatrix(triple)

        self.all_User.update(self.user)
        self.all_Item.update(self.item)
        for entry in self.testData:
            userId, itemId, rating = entry
            # order the user
            if not self.user.has_key(userId):
                self.all_User[userId] = len(self.all_User)
            # order the item
            if not self.item.has_key(itemId):
                self.all_Item[itemId] = len(self.all_Item)

            if not self.testSet_u.has_key(userId):
                self.testSet_u[userId] = {}
            self.testSet_u[userId][itemId] = rating
            if not self.testSet_i.has_key(itemId):
                self.testSet_i[itemId] = {}
            self.testSet_i[itemId][userId] = rating




    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            n = self.row(u) > 0
            mean = 0

            if not self.containsUser(u):  # no data about current user in training set
                pass
            else:
                sum = float(self.row(u)[0].sum())
                try:
                    mean =  sum/ n[0].sum()
                except ZeroDivisionError:
                    mean = 0
            self.userMeans[u] = mean

    def __computeItemMean(self):
        for c in self.item:
            n = self.col(c) > 0
            mean = 0
            if not self.containsItem(c):  # no data about current user in training set
                pass
            else:
                sum = float(self.col(c)[0].sum())
                try:
                    mean = sum / n[0].sum()
                except ZeroDivisionError:
                    mean = 0
            self.itemMeans[c] = mean

    def getUserId(self,u):
        if self.user.has_key(u):
            return self.user[u]
        else:
            return -1

    def getItemId(self,i):
        if self.item.has_key(i):
            return self.item[i]
        else:
            return -1

    def trainingSize(self):
        return (self.trainingMatrix.size[0],self.trainingMatrix.size[1],len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        return self.trainingMatrix.contains(self.getUserId(u),self.getItemId(i))

    def containsUser(self,u):
        'whether user is in training set'
        return self.trainingMatrix.matrix_User.has_key(self.getUserId(u))

    def containsItem(self,i):
        'whether item is in training set'
        return self.trainingMatrix.matrix_Item.has_key(self.getItemId(i))

    def userRated(self,u):
        if self.trainingMatrix.matrix_User.has_key(self.getUserId(u)):
            itemIndex =  self.trainingMatrix.matrix_User[self.user[u]].keys()
            rating = self.trainingMatrix.matrix_User[self.user[u]].values()
            return (itemIndex,rating)
        return ([],[])

    def itemRated(self,i):
        if self.trainingMatrix.matrix_Item.has_key(self.getItemId(i)):
            userIndex = self.trainingMatrix.matrix_Item[self.item[i]].keys()
            rating = self.trainingMatrix.matrix_Item[self.item[i]].values()
            return (userIndex,rating)
        return ([],[])

    def row(self,u):
        return self.trainingMatrix.row(self.getUserId(u))

    def col(self,c):
        return self.trainingMatrix.col(self.getItemId(c))

    def sRow(self,u):
        return self.trainingMatrix.sRow(self.getUserId(u))

    def sCol(self,c):
        return self.trainingMatrix.sCol(self.getItemId(c))

    def rating(self,u,c):
        return self.trainingMatrix.elem(self.getUserId(u),self.getItemId(c))

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return self.trainingMatrix.elemCount()
