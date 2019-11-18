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
 ** pyast_dump.mli: dumping Python abstract syntax trees *)
open Pyast_sig
module F = Format

(** Printing basic types *)
val pp_identifier: F.formatter -> identifier -> unit
val pp_num:        F.formatter -> number -> unit
val pp_str:        F.formatter -> string -> unit

(** Printing to formatters *)
val pp_modl: F.formatter -> 'a modl -> unit
val pp_stmt: F.formatter -> 'a stmt -> unit
val pp_expr: F.formatter -> 'a expr -> unit

(** Printing to stdout *)
val pp_print_modl: F.formatter -> 'a modl -> unit
val pp_print_stmt: F.formatter -> 'a stmt -> unit
val pp_print_expr: F.formatter -> 'a expr -> unit
val print_modl: 'a modl -> unit
val print_stmt: 'a stmt -> unit
val print_expr: 'a expr -> unit

(** Generation of strings *)
val dump_modl: 'a modl -> string
val dump_stmt: 'a stmt -> string
val dump_expr: 'a expr -> string
