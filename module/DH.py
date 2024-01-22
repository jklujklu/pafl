#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jklujklu
@contact:jklujklu@126.com
@version: 1.0.0
@license: Apache Licence
@file: DH.py
@time: 2024/1/11 14:18
"""
from diffiehellman import DiffieHellman


class KA:
    """Generates public and private keys and computes the shared key.
    """

    @staticmethod
    def gen() -> tuple:
        """Generates Diffie-Hellman public and private keys.

        Returns:
            Tuple[PublicKey, PrivateKey]: the public and private keys.
        """

        dh = DiffieHellman()
        pub_key, priv_key = dh.get_public_key(), dh.get_private_key()

        return pub_key, priv_key

    @staticmethod
    def agree(priv_key: bytes, pub_key: bytes) -> bytes:
        """Generates the shared key of two users, and produce 256 bit digest of the shared key.

        Args:
            priv_key (bytes): the private key of one user.
            pub_key (bytes): the public key of the other user.

        Returns:
            bytes: the 256 bit shared key of the two users.
        """
        dh = DiffieHellman()

        dh.set_private_key(priv_key)
        shared_key = dh.generate_shared_key(pub_key)


        return shared_key


if __name__ == '__main__':
    (p1, s1) = KA.gen()
    (p2, s2) = KA.gen()
    print(KA.agree(s1, p2) == KA.agree(s2, p1))
    print((2 ** 127) - 1)
