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
 ** adom_apron.ml: implementation of functors adding zone information
 **)
open Ir_sig
open Adom_sig

(** bnd_expr *)
val expr_to_bnd_expr: expr -> bnd_expr

(** zone domain constructor *)
module MakeId: ABST_DOMAIN_NB_D -> ABST_DOMAIN_ZONE_NB
module MakeZone: ABST_DOMAIN_NB_D -> ABST_DOMAIN_ZONE_NB
