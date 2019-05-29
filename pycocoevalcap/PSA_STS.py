__author__ = 'Qignzhong'

import os
import json
import pickle
import numpy as np

from collections import defaultdict
from requests import get


class COCOEvalCap:
    def __init__(self, pathToData, refName, candName, dfMode = "coco-val-df"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode
        self._url = "http://swoogle.umbc.edu/SimService/GetSimilarity"
        if self._dfMode != 'corpus':
            with open('./data/coco-val-df.p', 'r') as f:
                self._df_file = pickle.load(f)

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson(candName, num=10):
            path_to_cand_file = os.path.join(self._pathToData, candName)
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            res = defaultdict(list)

            for id_cap in cand_list:
                res[id_cap['image_id']].extend(id_cap['captions'])

            return res

        print 'Loading Data...'
        res = readJson(self._candName)
        ratio = {}
        for im_id in res.keys():
            print ('number of images: %d\n')%(len(ratio))
            cov = np.zeros([10, 10])
            for i in range(10):
                for j in range(i, 10):
                    new_gts = {}
                    new_res = {}
                    new_res[im_id] = [{'caption': res[im_id][i]}]
                    new_gts[im_id] = [{'caption':res[im_id][j]}]
                    score = get(self._url,
                                params={'operation':'api','phrase1':res[im_id][i],
                                'phrase2':res[im_id][j],'type':'relation','corpus':'webbase'})
                    while score.text == '':
                        score = get(self._url,
                                    params={'operation': 'api', 'phrase1': res[im_id][i],
                                            'phrase2': res[im_id][j], 'type': 'relation', 'corpus': 'webbase'})
                    # print res[im_id][i] + '  ' + res[im_id][j]
                    # print score.text
                    cov[i, j] = float(score.text.strip())
                    cov[j,i] = cov[i, j]
            u, s, v = np.linalg.svd(cov)
            r = max(s) / s.sum()
            print('ratio=%.5f\n')%(r)
            ratio[im_id] = r
            if len(ratio) == 5000:
                break

        self.eval = ratio

    def setEval(self, score, method):
        self.eval[method] = score

