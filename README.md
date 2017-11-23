# CombiLP

## About

This repository contains a re-implementation of the CombiLP algorithm in Python.
The original implementation was done by Bogdan Savchynskyy and has been
[integrated into opengm][opengm]. I rewrote the whole C++ code base to clean
everything up and implemented some major improvements (see references).

Before working on this port to Python I added functionality to the opengm code
myself. Unfortunately, this tight integration into opengm came with a price and
during development quick tests almost always resulted in a huge amount of
boilerplate code. At some point I started working on a complete independent
re-implementation. Note that performance is not a goal and this is all about
simplicity and maintainability. It should be behave more or less like the C++
implementation, though (see “Was has changed?”).

## References

 1. Savchynskyy, B., Kappes, J. H., Swoboda, P., & Schnörr, C. (2013). Global
    MAP-optimality by shrinking the combinatorial search area with convex
    relaxation. In Advances in Neural Information Processing Systems (pp.
    1950-1958).
 2. Haller, S., Swoboda, P., & Savchynskyy, B. (2018). Exact MAP-Inference by
    Confining Combinatorial Search with LP Relaxation. (To be published in AAAI
    18.)

Note that this is the not the code used the latest paper, but it should be
comparable and much easier to use and hack on, see next section.

## What has changed?

Compared to the [original implementation][combilp-opengm] many things have
changed. The whole logic has be completely rewritten in Python. The subsolvers
(that perform the "real" work) are still C/C++ libraries so their performance is
not affected. The overall pipeline has not changed much, so all the critical
characteristics (number of CombiLP-iterations, sizes of the ILP subproblems,
etc.) are more or less identical as in the original implementation.

As the reparametrization is currently implemented in Python it is much much
slower than before. It would be quite easy to accelerate everything using Cython
or similar techniques. If you look at the summary of the elapsed running time,
chances are very high that you are only interested in the time spent computing
ILP solutions.

I do not have any plans to speed up the reparametrization right now, as in a
serious implementation this time would be almost zero and the real speed-ups
come from the reduced size of the ILP suproblem.

## Requirements

  - Python 3
  - numpy
  - opengm (optional: to load opengm hdf5 model files)
  - subsolvers you want to use (most of the included as submodule, CPLEX is
    optional dependency)

[opengm]: https://github.com/opengm/opengm
[combilp-opengm]: https://github.com/fgrsnau/opengm/branch/dev

<!-- vim: set ts=2 sts=2 sw=2 et tw=80 fo+=a spell spl=en: -->
