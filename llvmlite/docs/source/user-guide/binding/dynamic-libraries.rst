=============================
Dynamic libraries and symbols
=============================

.. currentmodule:: llvmlite.binding

These functions tell LLVM how to resolve external symbols 
referred from compiled LLVM code.


* .. function:: add_symbol(name, address)

     Register the *address* of global symbol *name*, for use from 
     LLVM-compiled functions.


* .. function:: address_of_symbol(name)

     Get the in-process address of symbol *name*. An integer is 
     returned, or ``None`` if the symbol is not found.


* .. function:: load_library_permanently(filename)

     Load an external shared library. *filename* is the path to the
     shared library file.
     
