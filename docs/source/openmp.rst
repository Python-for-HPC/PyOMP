OpenMP support
==============

The following tables show supported OpenMP directives and the support status of
their clauses.

.. note::
   ✅ = supported;
   ❌ = unsupported;
   🔶 = partial support


.. csv-table::
   
   ,
   **barrier**,

.. csv-table::

   ,
   **critical**,

.. csv-table::

   ,allocate,collapse,firstprivate,lastprivate,linear,nowait,order,ordered,private,reduction,schedule
   **for**,❌,✅,✅,✅,❌,❌,❌,❌,✅,🔶,❌

.. csv-table::
   
   ,allocate,copyin,default,firstprivate,if,num_threads,private,proc_bind,reduction,shared
   **parallel**,❌,❌,✅,✅,✅,✅,✅,❌,🔶,✅

.. csv-table::
   
   ,*See clauses for* **parallel** *and* **for** directives
   **parallel for**,

.. csv-table::

   ,allocate,copyprivate,firstprivate,nowait,private
   **single**,❌,❌,❌,❌,❌

.. csv-table::

   ,affinity,allocate,default,detach,if,in_reduction,final,firstprivate,mergeable,priority,private,shared,untied
   **task**,❌,❌,✅,❌,❌,❌,❌,✅,❌,❌,✅,✅,❌

.. csv-table::

   ,depend,nowait
   **taskwait**,❌,❌

.. csv-table::

   ,allocate,defaultmap,depend,device,firstprivate,has_device_addr,if,in_reduction,is_device_ptr,map,nowait,private,thread_limit,uses_allocators
   **target**,❌,❌,❌,✅,✅,❌,❌,❌,❌,✅,❌,✅,✅,❌

.. csv-table::

   ,private,firstprivate,shared,reduction,default,num_teams,thread_limit
   **teams**,✅,✅,✅,🔶,✅,✅,✅

.. csv-table::

   ,allocate,collapse,dist_schedule,firstprivate,lastprivate,order,private
   **distribute**,❌,❌,❌,✅,✅,❌,✅

.. csv-table::

   ,*See clauses for* **teams** *and* **distribute** directives
   **teams distribute**,

.. csv-table::

   ,*See clauses for* **teams** *and* **teams** directives
   **target teams**,


.. csv-table::

   ,device,if,map,use_device_ptr,use_device_addr
   **target data**,✅,❌,✅,❌,❌
   
.. csv-table::

   ,depend,device,if,map,nowait
   **target enter data**,❌,✅,❌,✅,❌

.. csv-table::

   ,*See clauses for the* **target enter data** directive
   **target exit data**,

.. csv-table::

   ,nowait,depend,device,from,if,to
   **target update**,❌,❌,✅,✅,❌,✅

.. csv-table::

   ,*See clauses for* **target** *and* **teams distribute** directives
   **target teams distribute**,

.. csv-table::

   ,*See clauses for* **distribute** *and* **parallel for** directives
   **distribute parallel for**,

.. csv-table::

   ,*See clauses for* **target** *and* **teams** *and* **distribute parallel for** directives
   **target teams distribute parallel for**,
