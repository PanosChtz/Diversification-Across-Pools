# Diversification-Across-Pools

This tool automates the decision for an "active" cryptocurrency miner, who owns either commercial off-the-shelf hardware (e.g. CPUs/GPUs) or application-specific integrated circuit hardware (ASICs). IT allows them to optimally distribute their computational power over multiple pools and PoW cryptocurrencies (i.e build a mining portfolio), taking into account their risk aversion levels. Our tool allows miners to maximize their risk-adjusted earnings by diversifying across multiple mining pools and thus enhancing PoW decentralization.

# Prerequisites
Python3.x, numpy, scipy

# Usage
Tool is made of 3 different versions for computing the optimal hash power allocation of the miner.
Version 1 (function solvePools): Miner on a single cryptocurrency
Version 2 (function solvePoolsMultiCurr): Miner on multiple cryptocurrencies with same PoW algorithms
Version 3 (function solvePoolsMultiAlg): Miner on multiple cryptocurrencies with different PoW algorithms
