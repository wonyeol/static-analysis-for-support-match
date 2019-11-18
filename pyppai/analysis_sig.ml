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
 ** analysis_sig.ml: analysis options and general runtime related data-types *)

(** Analysis options *)

(* What kind of inputs *)
type analysis_input =
  | AI_pyfile of string   (* a python file name *)
  | AI_pystring of string (* a python program as a string *)

(* Numerical domains *)
type ad_num = AD_box | AD_oct | AD_pol

(* Options vectors *)
type analysis_opts =
    { (* which numerical domain to use *)
      ao_do_num:    ad_num option;
      (* where the input should be read *)
      ao_input:     analysis_input;
      (* whether to dump iterator debug information or not *)
      ao_debug_it:  bool;
      (* whether to have threshold widening *)
      ao_wid_thr:   bool;
      (* whether to activate the zone abstraction *)
      ao_zone:      bool;
      (* whether to simplify boolean expressions in assume *)
      ao_sim_assm:  bool }
