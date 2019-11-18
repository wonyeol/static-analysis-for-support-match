(** pyppai: basic abstract interpreter for python probabilistic programs
 **
 ** GNU General Public License
 ** 
 ** Authors:
 **  Wonyeol Lee, KAIST
 **  Xavier Rival, INRIA Paris
 **  Hongseok Yang, KAIST
 **  Hangyeol Yu, KAIST
 **
 ** Copyright (c) 2019 KAIST and INRIA Paris
 ** 
 ** ai_make.mli: construction of a static analyzer *)
open Ai_sig

(** Global switch for iterator debugging information *)
val debug: bool ref

(** Global switch for widening with thresholds *)
val wid_thr: bool ref

(** Global switch for simplifying expressions during the analysis of Assume *)
val sim_assm: bool ref

(** Generic functor to create analyzers *)
module MakeAnalysis: ANALYZER
