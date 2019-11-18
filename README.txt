STRUCTURE OF THIS README FILE:

   1. SUMMARY
   2. DEPENDENCES AND COMPILATION
   3. STANDALONE ANALYSIS
   4. BATCH ANALYSIS
   5. RESULTS IN OUR PAPER
   6. A QUICK OVERVIEW OF THE SOURCE CODE



1. SUMMARY

This file describes the options of both the standalone mode of the analysis
(to run it on a single file or on a single model-guide pair) and the batch
mode of the analysis (to run it on a series of test cases).

To proceed to the evaluation of the analysis on the supplied test cases,
one should simply refer to parts 2, 4, and 5 of this readme file.



2. DEPENDENCES AND COMPILATION

Before compiling the analyser, the following dependences should be satisfied:

 - ocaml     (version 4.07 or later)
 - opam      (opam is not completely mandatory but is strongly recommended to
              install the dependencies below)
 - obuild    (version 0.1.10 or later)
 - pyml      (version 20190626 or later)
 - stdcompat (version 11 or later)
 - apron     (version 20160125 or later)

There is no architecture dependency, provided the above packages can be
installed.

To compile everything, simply type the following command:

  make

or, alternatively,

  make execs

This will produce the following executables:

  pyppai.exe        # standalone analyser
  batch.exe         # batch mode to run the analyser on series of test cases



3. STANDALONE ANALYSIS

The analysis executable is pyppai.exe.
There are two execution modes:
 - analysis of a single python file (mostly useful to browse invariants):
        ./pyppai.exe prog.py
 - analysis for the verification of a model/guide pair:
        ./pyppai.exe -model m.py -guide g.py

The options fall in four categories:

a. Definition of a model and guide pair (for the verification of such a
   pair, both model and guide should be given; it is not possible to
 -model ARG.py  sets the model to consider in a model/guide pair
                (ARG.py should be a valid Python file)
 -guide ARG.py  sets the guide to consider in a model/guide pair
                (ARG.py should be a valid Python file)

b. Choice of the abstract domain:
 -ai-box    use of the interval domain for numerical constraints
            (conjunctions of constraints of the form a<=x<=b)
 -ai-oct    use of the octagon domain for numerical constraints
            (conjunctions of constraints of the form +/-x+/-y<=c)
 -ai-pol    use of the convex polyhedra domain for numerical constraints
            (conjunctions of linear inequalities with rational coefficients)
 -zone-on   activation of the zone domain for the description of tensor regions
 -zone-off  deactivation of the aforementioned tensor domain

c. Twicking the analysis level of precision in widening and transfer
   functions:
 -thr-on    activation of the widening with thresholds
            (the analysis of loops should be more precise, but may be slower)
 -thr-off   de-activation of the widening with thresholds
            (the analysis of loops should converge more quickly, but may be
            less precise)
 -sim-assm-on   activation of arithmetic expression simplifications
 -sim-assm-off  deactivation of arithmetic expression simplifications

d. Verbosity level:
 -q         quiet mode, will print very few debug information and invariants
 -v         verbose mode, will print extensive debug information and invariants



4. BATCH ANALYSIS

The batch analysis executable is batch.exe. It runs in sequences series of
tests defined in the file pyppai/data.ml (see description below) and prints a
synthetic summary of the results. It is possible to modify pyppai/data.ml to
include different series of test cases.
There are two execution modes:
 - analysis of the Pyro test suite (39 model/guide pairs):
        ./batch.exe suite
 - analysis of the Pyro examples (12 original model/guide pairs):
        ./batch.exe examples

The options fall in two categories:

a. Timeout:
 -to <SECS>  sets the timeout for each test to SECS (integer, in seconds)

b. Verbosity level:
 -q         quiet mode, will print very few debug information and invariants
 -v         verbose mode, will print extensive debug information and invariants

The batch.exe tool will output the results of each of the analysis in the
series of model/guide pairs it is instructed to analyse, and will produce
a table of results with the following columns:

- Num:     index of the test
- Match:   whether the result of the analysis is conclusive or not
    yes     conclusive analysis (the model/guide pair is either appropriately
            considered valid or invalid)
     no     inconclusive analysis (either due to a crash or to the conservative
            rejection of a correct pair)
- Expect:  actual characterization of the model/guide pair, that we would
           expect the analysis to find
    Valid   the model/guide pair should be valid
  Invalid   the model/guide pair should be invalid (e.g., sampled dimensions
            are not consistent)
      ERR   the model/guide pair may produce errors (e.g., due to plates
            or broadcasting)
- Result:  result produced by the analysis
    Valid   the model/guide pair is reported as valid
  Invalid   the model/guide pair is reported as possibly invalid (e.g.,
            sampled dimensions are not proved consistent)
      ERR   the model/guide pair is reported as possibly producing errors
            (e.g., due to plates or broadcasting)
    Crash   the analysis failed due to an error in the analyser itself
            (likely due to unsupported cases)
- Time (s) time taken by the analysis, in seconds
- Test     description of the test case



5. RESULTS IN OUR PAPER

To reproduce the results reported in our paper (Table 2), run the
following commands:

  ./batch.exe examples -q -to 100
  ./batch.exe suite -q -to 100

Expected results are as follows:

 - examples:
     ...
     SUMMARY:
     TESTS:   12
     VALID:   9
     INVALID: 3
     ERR:     0
     CRASH:   0
     TO:      0  
 - suite:
     ...
     SUMMARY:
     TESTS:   39
     VALID:   20
     INVALID: 3
     ERR:     8
     CRASH:   8
     TO:      0

We observe that it takes much longer to run the analysis in virtual
machines, so you might obtain analysis times much longer than the
times reported in Table 2. If you obtain different results from the
above, try using a larger value of timeout (e.g., -to 200).

We remark that the green boxes in #Diff column of Table 2 (left) and
Valid? column of Table 2 (right) correspond to the following
model-guide pairs:

 - "suite, n-wplate, nested_plate_plate_dim_error_1" in suite
 - "suite, n-wplate, nested_plate_plate_dim_error_3" in suite
 - "BR example, original" in examples
 - "LDA example, original" in examples

More details on the model-guide pairs are described below: 

 - For Table 2 (left), the two highlighted model-guide pairs in the
   "Nested with-plates" category correspond to the pairs with name
   "suite, n-wplate, nested_plate_plate_dim_error_1" and "..._error_3"
   (i.e., with number 26 and 28) in the artifact. While they are
   documented in Pyro suite as invalid pairs, we manually checked that
   they are in fact valid pairs (as described in the paper). Hence, we
   set "Expect" value for the pairs to "Valid", and our analysis
   outputs "Valid" for both pairs.

 - For Table 2 (right), the two highlighted examples are BR and LDA;
   for each for these two, we present the original version (which is
   reported as "Invalid" as expected) and the edited versions where we
   try to solve the problem (note that, in the case of LDA, the first
   edited version is still reported as incorrect, as it should be).



6. A QUICK OVERVIEW OF THE SOURCE CODE

We now describe the contents of the files in this directory and in its
sub-directories:

  README.txt     this note
  makefile       the makefile to compile and run the main tests
  pyppai.obuild  the configuration for the OCaml build system obuild
  pyppai/        the source code of the analyser and batch system
  tests/         all test cases (our own and taken from Pyro examples and
                 Pyro test suite)

The files in the pyppai/ directory have the following roles:

- Entry point for the bach mode:
 entry_batch.ml

- Data files for the batch mode (list of Pyro test suite and examples
  programs):
 data.ml

- Entry point for the analysis:
 entry_analysis.ml

- Preparation of the analysis initial state and exploitation of results:
 main.ml             main.mli

- Abstract interpreter, related signature files, and analysis launch:
 analysis_sig.ml     ai_make.ml          ai_make.mli         ai_sig.ml

- Abstract domain, signatures of all modules related to the abstract
  domains:
 adom_sig.ml

- Abstract domain, abstraction of the properties of models and guides
  and reduction with numerical/shape constraints (so called "fibered
  layer" as the topology of abstract states is highly dynamic):
 adom_fib.ml         adom_fib.mli

- Abstract domain, abstraction of tensor zones (some for of shape
  properties in higher dimensions multi-arrays):
 adom_zone.ml        adom_zone.mli

- Abstract domain, numerical constraints:
 adom_apron.ml       adom_apron.mli

- Abstract domain, factoristion of a bottom element:
 adom_bot.ml         adom_bot.mli

- Abstract domain, abstraction of distributions:
 adom_distty.ml      adom_distty.mli

- Intermediate representation:
 ir_cast.ml          ir_cast.mli         ir_sig.ml           ir_util.ml
 ir_util.mli         irty_sig.ml         irty_util.ml        irty_util.mli

- Front-end and interfacing with pyml:
 pyast_cast.ml       pyast_cast.mli      pyast_dump.ml       pyast_dump.mli
 pyast_sig.ml        pyast_util.ml       pyast_util.mli      pyobj_util.ml
 pyobj_util.mli

- Library and auxiliary functions:
 lib.ml

The names of the test-cases in tests/ are self-explanatory.

