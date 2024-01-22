#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: compare_time.py
@time: 2024/1/11 21:02
"""
import argparse
import math
import sys
import time

from loguru import logger

from entities.server import Server, TA
import warnings

warnings.filterwarnings("ignore")

for function in ['pafl', 'krum', 'avg', 'foolsgold', 'median', 'trimmed']:
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    logger.add('./log/time_{}.log'.format(function), level="ERROR")
    for user in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f'Function: {function}, User: {user}')
        user_gradients_sum = {}

        ml_server = Server(0.01)
        s = TA(user, math.ceil(math.log(user, 2)), 0)

        s.init_clients()
        s.collect_shares()

        s.collect_gradients(ml_server.get_model(), function=function)

        start = time.time()
        new_model = s.aggregate(function=function)
        logger.error(f'User: {user} | Used Time: {int((time.time() - start) * 1000)}ms')

        ml_server.set_model(new_model)
        ml_server.local_val()
        s.clear_gradients()
