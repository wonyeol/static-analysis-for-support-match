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
 ** irty_util.mli: utilities for types of ir *)
open Ir_sig
open Irty_sig

(** to_string *)
val num_ty_to_string: num_ty -> string
val tensor_size_ty_to_string: tensor_size_ty -> string
val plate_ty_to_string: plate_ty -> string
val range_ty_to_string: range_ty -> string
val distr_ty_to_string: distr_ty -> string
val fun_ty_to_string: fun_ty -> string
val exp_ty_to_string: exp_ty -> string
val vtyp_to_string: vtyp -> string
val broadcast_info_to_string: broadcast_info -> string
val dist_size_ty_to_string: dist_size_ty -> string

(** tensor_size_ty *)
exception Broadcast_failure (* signals the must failure of broadcast *)
val tensor_size_ty_leq: tensor_size_ty -> tensor_size_ty -> bool
val tensor_size_ty_join: tensor_size_ty -> tensor_size_ty -> tensor_size_ty
val tensor_size_ty_concat: tensor_size_ty -> tensor_size_ty -> tensor_size_ty

(** num_ty *)
val num_ty_leq: num_ty -> num_ty -> bool
val num_ty_join: num_ty -> num_ty -> num_ty

(** plate_ty *)
val plate_ty_leq: plate_ty -> plate_ty -> bool
val plate_ty_join: plate_ty -> plate_ty -> plate_ty

(** range_ty *)
val range_ty_leq: range_ty -> range_ty -> bool
val range_ty_join: range_ty -> range_ty -> range_ty

(** distr_ty *)
val distr_ty_leq: distr_ty -> distr_ty -> bool
val distr_ty_join: distr_ty -> distr_ty -> distr_ty

(** fun_ty *)
val fun_ty_leq: fun_ty -> fun_ty -> bool
val fun_ty_join: fun_ty -> fun_ty -> fun_ty
val fun_ty_apply: fun_ty -> tensor_size_ty -> tensor_size_ty

(** exp_ty *)                                                       
val get_exp_ty:    (idtf -> exp_ty) -> expr -> exp_ty
val is_int_exp:    (idtf -> exp_ty) -> expr -> bool
val is_real_exp:   (idtf -> exp_ty) -> expr -> bool
val is_tensor_exp: (idtf -> exp_ty) -> expr -> bool
val is_fun_exp:    (idtf -> exp_ty) -> expr -> bool
(* check whether a given type can be broadcasted to a tensor *)
val is_broadcastable_exp_ty : exp_ty -> bool
(* select tensor types from a given list of expression types, and broadcast the selected types *)
val exp_ty_do_broadcast: exp_ty list -> tensor_size_ty

(** vtyp *)
val vtyp_leq: vtyp -> vtyp -> bool

(** broadcast_info *)
val next_dim_from_broadcast_info: broadcast_info -> int
  
(** dist_size_ty *)
module type DistSizeTy =
  sig
    val get_ty: exp_ty list -> dist_kind -> dist_size_ty option
    val apply_trans_l: dist_trans list -> dist_size_ty option -> dist_size_ty option
    val apply_broadcast: broadcast_info -> dist_size_ty option -> dist_size_ty option
    val get_logprob_ts: exp_ty -> dist_size_ty option -> tensor_size_ty
  end
module DistSize : DistSizeTy 
