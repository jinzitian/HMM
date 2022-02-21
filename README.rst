=================
HMM
=================

A fast Python implementation of HMM

Installation
============
To install:

.. code-block:: bash

    $ pip install hmm-tool

Quickstart
==========

.. code-block:: python

    from hmmtool import HMM
    import numpy as np
    
    ### 3 states, 6 observations:
    ### states values:0/1/2, observations values:0/1/2/3/4/5 
    hmm = HMM(3,6)
	
    #train
    s1 = np.random.randint(6,size = 60)
    s2 = np.random.randint(6,size = 40)
    hmm.add_data([s1,s2])
    hmm.train(maxStep=50,delta=0.001)
	
    #get params
    print(hmm.pi)
    print(hmm.A)
    print(hmm.B)
	
    #predict
    #random data
    s3 = np.random.randint(6,size = 10)
    s4 = np.random.randint(6,size = 10)
	
    #multi inputs:[[o1,o2,o3,...,ot1],[o1,o2,o3,...,ot2]]
    #output: [prob1, prob2]
    print(hmm.estimate_prob([s3,s4]))
    #output: [(prob1, [s1,s2,s3,...,st1]), (prob2, [s1,s2,s3,...,st2])]
    print(hmm.decode([s3,s4]))
	  
    
	
