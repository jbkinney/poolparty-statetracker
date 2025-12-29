API Reference
=============

Core Classes
------------

Counter
~~~~~~~

.. autoclass:: statecounter.Counter
   :members:
   :special-members: __init__, __iter__, __getitem__

Manager
~~~~~~~~~~~~~~

.. autoclass:: statecounter.Manager
   :members:
   :special-members: __init__, __enter__, __exit__

Operation
~~~~~~~~~~~~~~~~

.. autoclass:: statecounter.Operation
   :members:

Exceptions
----------

.. autoclass:: statecounter.ConflictingStateAssignmentError

Counter Operations (Ops)
--------------------------

ProductOp
~~~~~~~~~~~~

.. autoclass:: statecounter.ProductOp
   :members:

.. autofunction:: statecounter.product_counters

StackOp
~~~~~~~

.. autoclass:: statecounter.StackOp
   :members:

.. autofunction:: statecounter.sum_counters

SynchronizeOp
~~~~~~~~~~~~~~~

.. autoclass:: statecounter.SynchronizeOp
   :members:

.. autofunction:: statecounter.synchronize_counters

SliceOp
~~~~~~~~~

.. autoclass:: statecounter.SliceOp
   :members:

.. autofunction:: statecounter.slice_counter

RepeatOp
~~~~~~~~~~

.. autoclass:: statecounter.RepeatOp
   :members:

.. autofunction:: statecounter.repeat_counter

ShuffleOp
~~~~~~~~~~~

.. autoclass:: statecounter.ShuffleOp
   :members:

.. autofunction:: statecounter.shuffle_counter

InterleaveOp
~~~~~~~~~~~~~~

.. autoclass:: statecounter.InterleaveOp
   :members:

.. autofunction:: statecounter.interleave_counters

PassthroughOp
~~~~~~~~~~~~~~~

.. autoclass:: statecounter.PassthroughOp
   :members:

.. autofunction:: statecounter.passthrough_counter

Utility Functions
-----------------

.. autofunction:: statecounter.split_counter
