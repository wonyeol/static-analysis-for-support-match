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
 ** ir_util.ml: utilities over the ir form, including pretty-printing *)
open Ir_sig
open Irty_sig

(** to_{int,string}_opt *)
val expr_to_int_opt: expr -> int option
val expr_to_string_opt: expr -> string option

(** Checking various properties of statements *)
val contains_continue: stmt -> bool
val contains_break: stmt -> bool
(* val contains_return: stmt -> bool *)

(** dist *)
(* yes means true, but false means don't know *)
val dist_kind_support_subseteq: dist_kind -> dist_kind -> bool

(** Functions for modifying and simplifying expressions. *)
(* simplify_exp's argument performs an approximate type inference 
 * of an expression *)
val simplify_exp: (expr -> exp_ty) -> expr -> expr

(** ***************)
(** string, print *)
(** ***************)
(** Constants into strings *)
val uop_to_string: uop -> string
val bop_to_string: bop -> string
val cop_to_string: cop -> string
val dist_kind_to_string:  dist_kind -> string
val dist_trans_to_string: dist_trans -> string

(** Functions to print into buffers *)
val buf_dist:   Buffer.t -> dist -> unit
val buf_number: Buffer.t -> number -> unit
val buf_expr:   Buffer.t -> expr -> unit
val buf_acmd:   Buffer.t -> acmd -> unit
val buf_stmt:   Buffer.t -> stmt -> unit
val buf_block:  Buffer.t -> block -> unit
val buf_prog:   Buffer.t -> prog -> unit

(** Conversion to strings *)
val number_to_string: number -> string
val expr_to_string:   expr -> string
val acmd_to_string:   acmd -> string
val prog_to_string:   prog -> string

(** Pretty-printing on channels *)
val pp_number:    out_channel -> number -> unit
val pp_expr:      out_channel -> expr -> unit
val pp_expr_list: out_channel -> expr list -> unit
val pp_acmd:      out_channel -> acmd -> unit
val pp_stmt:      out_channel -> stmt -> unit
val pp_block:     out_channel -> block -> unit
val pp_prog:      out_channel -> prog -> unit
