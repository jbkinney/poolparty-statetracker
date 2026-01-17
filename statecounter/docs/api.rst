API Reference
=============

This page provides complete API documentation for all public classes and functions
in StateCounter, automatically generated from source code docstrings.

.. contents:: On this page
   :local:
   :depth: 2

Core Classes
------------

Counter
~~~~~~~

The main class for creating and composing counters.

.. autoclass:: statecounter.Counter
   :members:
   :special-members: __init__, __iter__, __getitem__

Manager
~~~~~~~

Context manager that must wrap all counter operations.

.. autoclass:: statecounter.Manager
   :members:
   :special-members: __init__, __enter__, __exit__

Operation
~~~~~~~~~

Abstract base class for all counter operations.

.. autoclass:: statecounter.Operation
   :members:

Exceptions
----------

.. autoexception:: statecounter.ConflictingStateAssignmentError
   :show-inheritance:

Counter Operation Functions
---------------------------

These functions create new counters by combining or transforming existing counters.

Product Operations
~~~~~~~~~~~~~~~~~~

.. autofunction:: statecounter.product

.. autofunction:: statecounter.ordered_product

.. autofunction:: statecounter.set_product_order_mode

.. autofunction:: statecounter.get_product_order_mode

Stack (Disjoint Union)
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statecounter.stack

Synchronize
~~~~~~~~~~~

.. autofunction:: statecounter.sync

Slice
~~~~~

.. autofunction:: statecounter.slice

Repeat
~~~~~~

.. autofunction:: statecounter.repeat

Shuffle
~~~~~~~

.. autofunction:: statecounter.shuffle

Sample
~~~~~~

.. autofunction:: statecounter.sample

Split
~~~~~

.. autofunction:: statecounter.split

Interleave
~~~~~~~~~~

.. autofunction:: statecounter.interleave

Passthrough
~~~~~~~~~~~

.. autofunction:: statecounter.passthrough

Operation Classes
-----------------

These are the underlying operation classes used internally. Most users will use
the convenience functions above instead of instantiating these directly.

ProductOp
~~~~~~~~~

.. autoclass:: statecounter.ProductOp
   :members:
   :show-inheritance:

StackOp
~~~~~~~

.. autoclass:: statecounter.StackOp
   :members:
   :show-inheritance:

SyncOp
~~~~~~

.. autoclass:: statecounter.SyncOp
   :members:
   :show-inheritance:

SliceOp
~~~~~~~

.. autoclass:: statecounter.SliceOp
   :members:
   :show-inheritance:

RepeatOp
~~~~~~~~

.. autoclass:: statecounter.RepeatOp
   :members:
   :show-inheritance:

ShuffleOp
~~~~~~~~~

.. autoclass:: statecounter.ShuffleOp
   :members:
   :show-inheritance:

SampleOp
~~~~~~~~

.. autoclass:: statecounter.SampleOp
   :members:
   :show-inheritance:

InterleaveOp
~~~~~~~~~~~~

.. autoclass:: statecounter.InterleaveOp
   :members:
   :show-inheritance:

PassthroughOp
~~~~~~~~~~~~~

.. autoclass:: statecounter.PassthroughOp
   :members:
   :show-inheritance:
