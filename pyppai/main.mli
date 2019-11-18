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
 ** main.ml: launching of the analysis (with options already parsed) *)
open Ir_sig
open Analysis_sig

(** Output *)
val output: bool ref

(** Default options *)
val aopts_default: analysis_opts

(** Parsing *)
val parse_code: int option -> analysis_input -> Ir_sig.prog

(** Expected outcome of a relational test case wrt. a default test *)
type test_oracle_r =
  | TOR_succeed (* the default test should succeed *)
  | TOR_fail    (* the default test should fail *)
  | TOR_error   (* must error *)

val string_of_test_oracle_r: test_oracle_r -> string

(** Master functions: *)
val start: bool -> analysis_opts list -> expr list -> expr list -> bool
val start_nr: bool -> analysis_opts -> expr list -> expr list -> bool
val start_r: bool -> analysis_opts -> analysis_opts -> test_oracle_r
  -> bool * string
val run_r: bool -> analysis_opts -> analysis_opts -> test_oracle_r
  -> bool option * bool
