#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: secureaggregation.py
@time: 2024/1/11 14:05
"""

import shamirs

ss = shamirs.shares(456, quantity=20, modulus=15485867, threshold=10)
print(ss)
bb = shamirs.interpolate(ss[5:], threshold=10)
print(bb)