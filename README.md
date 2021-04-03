# HDMM
Source code for [HDMM: The High-Dimensional Matrix Mechanism](http://www.vldb.org/pvldb/vol11/p1206-mckenna.pdf), which first appeared in VLDB 2018.

# Usage
HDMM is a mechanism for answering sets of linear queries under differential privacy with optimized accuracy.

**hdmm.templates** contains "strategy templates", which implement the various parameterizations used for strategy optimization.  These templates can be used to optimize the parameters of a particular type of strategy to tailor it to a given input workload. Note that **hdmm.matrix** and **hdmm.workload** contain classes to represent workloads and strategies in an implicit matrix form.  This code was copied from the [Ektelo repository](https://github.com/dpcomp-org/ektelo/) for convenience (to simplify dependencies).  

Here is one of the simplest use-cases of HDMM: to optimize a 1D range query workload over a domain of size 256 using the p-Identity strategy parameterization, and report the root-mean-squared-error of the resulting strategy.

```
>>> from hdmm import workload, templates, error
>>> W = workload.AllRange(256)
>>> pid = templates.PIdentity(16, 256)
>>> pid.optimize(W)
>>> A = pid.strategy()
>>> error.rootmse(W, A)
5.736437154199246
```

This is a relatively small workload.  HDMM really shines when running on larger multi-dimensional workloads.  Here, we optimize a 2D range query workload using a the Kronecker parameterization:

```
>>> from hdmm import workload, templates, error
>>> R = workload.AllRange(256)
>>> W = workload.Kronecker([R, R]) # All 2D Range queries
>>> kron = templates.KronPIdentity([16, 16], [256, 256])
>>> kron.optimize(W)
>>> A = kron.strategy()
>>> error.rootmse(W, A)
32.692354993620214
```

# Setup

Setup instructions are for an Ubuntu system.  First clone the repository, and add it to your PYTHONPATH by adding the following line to your .bashrc file:

```bash
export PYTHONPATH=/path/to/hdmm/src/
```

(Optional) now create a python virtual environment for HDMM as follows

```bash
$ mkdir ~/virtualenvs
$ python3 -m venv ~/virtualenvs/hdmm
$ source ~/virtualenvs/hdmm/bin/activate
```

And install the required dependencies:

```bash
$ pip install -r requirements.txt
```

Finally make sure the tests are passing

```bash
$ cd test/
$ nosetests
...................
----------------------------------------------------------------------
Ran 19 tests in 1.272s

OK
```


# Documentation

HDMM consists of 6 main files, which we describe here.  

## hdmm.matrix and hdmm.workload

These files are used to represent workload and strategy matrices implicitly.  This code has been copied from the ektelo repository for convenience.  Please refer to section 7 of [that paper](https://arxiv.org/pdf/1808.03555.pdf) for the design principles underlying these classes.  The relevant classes in **hdmm.matrix** are enumerated here:

* **EkteloMatrix**: General class for representing any matrix.  Parent of all other classes.  Can be instantiated with a numpy array or scipy sparse matrix.
* **Identity**: An n x n identity matrix
* **Ones**: An n x m matrix of ones
* **Weighted**: A class for multiplying a matrix by constant.
* **VStack**: A class for stacking matrices vertically (e.g., to append two or more subworkloads into one big workload)
* **Kronecker**: A class for constructing multi-dimensional matrices from 1D building blocks


**hdmm.workload** contains more convenience classes for constructing common workloads quickly.

* **Total**: The 1 x n matrix of ones
* **IdentityTotal**: The matrix [I; T] where I is Identity and T is Total
* **Prefix**: The n x n Prefix workload (range queries)
* **AllRange**: The workload of all range queries
* **Marginal**: The workload for a single marginal
* **Marginals**: THe workload for a collection of weighted marginal queries
* **DimKMarginals:** Helper function to compute workload of all marginals of a given dimensionality over a given domain (e.g., W = DimKMarginals([2,3,4,5,6,7], 3) returns workload for all 3-way marginals over a 6D domain)

There are other classes in these files that may be useful to some people, but the above list a good high-level overview and starting point.  

## hdmm.templates and hdmm.more_templates

This file contain code for *strategy templates*.  These templates encode strategies of a particular form, generally with a number of free parameters which can be optimized to produce a strategy tailored to a given workload.  These templates available are enumerated below:

* **PIdentity**: This template corresponds to $OPT_0$ (for Laplace noise)
* **Kronecker**: This template corresponds to $OPT_{\otimes}$
* **UnionKron**: This template corresponds to $OPT_{+}$
* **Marginals**: This template corresponds to $OPT_M$

Some of the templates (like PIdentity) perform optimization under the assumption Laplace noise, while others (like YuanConvex) perform optimization under the assumption of Gaussian noise.  Other templates (like Kronecker) are defind in terms of other templates, and can be used for either.  Finally, some templates (like Marginals) have an "approx" field which can be set to True for Gaussian noise (i.e., approximate differential privacy) or False for Laplace noise (i.e., pure differential privacy).  

A few useful helper functions for constructing general-purpose templates are listed below:

* **OPT0**: produces a different template depending on if approx=True or False.  If approx=False, the p parameter in the p-Identity strategy is set to a default value.
* **DefaultKron**: produces a Kronecker product template defined in terms of OPT0, and can be passed approx=True or False as well.
* **DefaultUnionKron**: produces a Union-of-Kronecker product template definded in terms of DefaultKron.  Also has an approx flag.

There are some other helper functions in **hdmm.templates** and **hdmm.more_templates** for advanced users.  

## hdmm.error

This file contains utilities for estimating error of a strategy on a workload.  Right now, this code only works for strategies produced for Laplace noise (i.e., $\delta = 0$).

## hdmm.mechanism

This file contains a class for actually running the end-to-end HDMM mechanism.  It requries a workload and a data vector and produces an estimated data vector that can be used to answer the workload.  There are certain variants and extensions to HDMM that can make it more scalable which are not implemented here.   This implementation assumes the data vector is sufficiently small that it can fit into memory and be operated on efficiently.  


## LICENSE

(C) Copyright 2019-2021, Tumult Labs Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

