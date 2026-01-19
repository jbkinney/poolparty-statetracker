API Reference
=============

This page provides complete API documentation for all public classes and functions
in StateTracker, automatically generated from source code docstrings.

.. contents:: On this page
   :local:
   :depth: 2

Core Classes
------------

State
~~~~~

The main class for creating and composing states.

.. autoclass:: statetracker.State
   :members:
   :special-members: __init__, __iter__, __getitem__

Manager
~~~~~~~

Context manager that must wrap all state operations.

.. autoclass:: statetracker.Manager
   :members:
   :special-members: __init__, __enter__, __exit__

Operation
~~~~~~~~~

Abstract base class for all state operations.

.. autoclass:: statetracker.Operation
   :members:

Exceptions
----------

.. autoexception:: statetracker.ConflictingValueAssignmentError
   :show-inheritance:

State Operation Functions
-------------------------

These functions create new states by combining or transforming existing states.

Product Operations
~~~~~~~~~~~~~~~~~~

.. autofunction:: statetracker.product

.. autofunction:: statetracker.ordered_product

.. autofunction:: statetracker.set_product_order_mode

.. autofunction:: statetracker.get_product_order_mode

Stack (Disjoint Union)
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statetracker.stack

Synchronize
~~~~~~~~~~~

.. autofunction:: statetracker.sync

Slice
~~~~~

.. autofunction:: statetracker.slice

Repeat
~~~~~~

.. autofunction:: statetracker.repeat

Shuffle
~~~~~~~

.. autofunction:: statetracker.shuffle

Sample
~~~~~~

.. autofunction:: statetracker.sample

Split
~~~~~

.. autofunction:: statetracker.split

Interleave
~~~~~~~~~~

.. autofunction:: statetracker.interleave

Passthrough
~~~~~~~~~~~

.. autofunction:: statetracker.passthrough

Operation Classes
-----------------

These are the underlying operation classes used internally. Most users will use
the convenience functions above instead of instantiating these directly.

ProductOp
~~~~~~~~~

.. autoclass:: statetracker.ProductOp
   :members:
   :show-inheritance:

StackOp
~~~~~~~

.. autoclass:: statetracker.StackOp
   :members:
   :show-inheritance:

SyncOp
~~~~~~

.. autoclass:: statetracker.SyncOp
   :members:
   :show-inheritance:

SliceOp
~~~~~~~

.. autoclass:: statetracker.SliceOp
   :members:
   :show-inheritance:

RepeatOp
~~~~~~~~

.. autoclass:: statetracker.RepeatOp
   :members:
   :show-inheritance:

ShuffleOp
~~~~~~~~~

.. autoclass:: statetracker.ShuffleOp
   :members:
   :show-inheritance:

SampleOp
~~~~~~~~

.. autoclass:: statetracker.SampleOp
   :members:
   :show-inheritance:

InterleaveOp
~~~~~~~~~~~~

.. autoclass:: statetracker.InterleaveOp
   :members:
   :show-inheritance:

PassthroughOp
~~~~~~~~~~~~~

.. autoclass:: statetracker.PassthroughOp
   :members:
   :show-inheritance:
