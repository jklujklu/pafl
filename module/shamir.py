#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: shamir.py
@time: 2024/1/11 14:11
"""
import pickle

import shamirs

# class Shamir:
#     """Shamir's t-out-of-n Secret Sharing.
#     """
#
#     @staticmethod
#     def ss(secret: int, t: int, n: int) -> list:
#         """Generates a set of shares.
#
#         Args:
#             secret (object): the secret to be split.
#             t (int): the threshold of being able to reconstruct the secret.
#             n (int): the number of the shares.
#
#         Returns:
#             list: a set of shares.
#         """
#         shares = shamirs.shares(secret, quantity=n, threshold=t)
#         return list(shares)
#
#     @staticmethod
#     def reco(shares: list, t: int):
#         secret = shamirs.interpolate(shares, threshold=t)
#         return secret
from secretsharing import SecretSharer

from module.DH import KA


class Shamir:
    """Shamir's t-out-of-n Secret Sharing.
    """

    @staticmethod
    def ss(secret: object, t: int, n: int) -> list:
        """Generates a set of shares.

        Args:
            secret (object): the secret to be split.
            t (int): the threshold of being able to reconstruct the secret.
            n (int): the number of the shares.

        Returns:
            list: a set of shares.
        """

        secret_bytes = pickle.dumps(secret)

        # convert bytes to hex
        secret_hex = secret_bytes.hex()

        shares = SecretSharer.split_secret(secret_hex, t, n)

        return shares

    @staticmethod
    def reco(shares: list):
        secret_hex = SecretSharer.recover_secret(shares)

        # convert hex to bytes
        secret_bytes = bytes.fromhex(secret_hex)

        secret = pickle.loads(secret_bytes)

        return secret


if __name__ == '__main__':
    a, b = KA.gen()
    print(Shamir.ss(b, 5, 10))
