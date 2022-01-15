===============================
DeepWalk
===============================

DeepWalk uses short random walks to learn representations for vertices in graphs.

Usage
-----

**Example Usage**
    ``$python deepwalk/__main__.py --input cora --output cora.embeddings``

Requirements
------------
gensim==4.1.2
matplotlib==3.3.4
networkx==2.5.1
pandas==1.1.5
scikit-learn==0.24.2
stellargraph==1.2.1
chardet==4.0.0
psutil==5.9.0

(may have to be independently installed) 
or `pip install -r requirements_core.txt` to install all dependencies


Citing
------
If you find DeepWalk useful in your research, we ask that you cite the following paper::

    @inproceedings{Perozzi:2014:DOL:2623330.2623732,
     author = {Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
     title = {DeepWalk: Online Learning of Social Representations},
     booktitle = {Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
     series = {KDD '14},
     year = {2014},
     isbn = {978-1-4503-2956-9},
     location = {New York, New York, USA},
     pages = {701--710},
     numpages = {10},
     url = {http://doi.acm.org/10.1145/2623330.2623732},
     doi = {10.1145/2623330.2623732},
     acmid = {2623732},
     publisher = {ACM},
     address = {New York, NY, USA},
     keywords = {deep learning, latent representations, learning with partial labels, network classification, online learning, social networks},
    } 

Misc
----

DeepWalk - Online learning of social representations.

* Free software: GPLv3 license

.. image:: https://badge.fury.io/py/deepwalk.png
    :target: http://badge.fury.io/py/deepwalk

.. image:: https://travis-ci.org/phanein/deepwalk.png?branch=master
        :target: https://travis-ci.org/phanein/deepwalk

.. image:: https://pypip.in/d/deepwalk/badge.png
        :target: https://pypi.python.org/pypi/deepwalk
