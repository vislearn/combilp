# CombiLP

## About

This repository contains a re-implementation of the CombiLP algorithm in Python.
The original implementation was done by Bogdan Savchynskyy and has been
[integrated into opengm][opengm]. I rewrote the whole C++ code base to clean
everything up and implemented some major improvements (see references).

You can find more information on the [project's website][website].

## References

 1. *B. Savchynskyy, J. Kappes, P. Swoboda, C. Schnörr.*
    **Global MAP-Optimality by Shrinking the Combinatorial Search Area with
    Convex Relaxation.**
    NIPS-2013.
 2. *S. Haller, P. Swoboda and B. Savchynskyy.*
    **Exact MAP-Inference by Confining Combinatorial Search with LP Relaxation.**
    Accepted to AAAI 2018.

Note that this is the not the code used the latest paper, but it should be
comparable and much easier to use and hack on, see next section.


## Requirements

  - Python 3
  - numpy
  - opengm (optional: to load opengm hdf5 model files)
  - subsolvers you want to use (most of the included as submodule, CPLEX is
    optional dependency)

[website]: https://fgrsnau.github.io/combilp
[opengm]: https://github.com/opengm/opengm

<!-- vim: set ts=2 sts=2 sw=2 et tw=80 fo+=a spell spl=en: -->
