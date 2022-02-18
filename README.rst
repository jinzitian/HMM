=================
AC Auto Mechine
=================

A fast Python implementation of Ac Auto Mechine

Installation
============
To install:

.. code-block:: bash

    $ pip install acAutoMechine

Quickstart
==========

.. code-block:: python

    from acAutoMechine import Ac_mechine
    
    ### usage one:
    actree = Ac_mechine()
    actree.add_keys('he')
    actree.add_keys('her')
    actree.add_keys('here')
    actree.build_actree()
    ### all match
    print(actree.match("he here her"))  
    ### long match
    print(actree.match_long("he here her"))  
    ### all match with match path
    print(actree.match("he here her", True))  
    ### long match with match path
    print(actree.match_long("he here her", True))  
    
	
