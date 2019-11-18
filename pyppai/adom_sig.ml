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
 ** adom_sig.ml: signature of abstract domain components
 **)
open Lib
open Ir_sig
open Irty_sig

(** Bottom exception *)
exception Bottom

(** Bound expressions *)
type bnd_expr =
  | BE_cst of int
  | BE_var of string
  | BE_var_minus_one of string

(** Signature for a general abstract domain *)

(** Domain without bottom *)
module type ABST_DOMAIN_NB =
  sig
    val module_name: string

    type t
    (* printing *)
    val buf_t:      Buffer.t -> t -> unit
    val to_string:  t -> string
    val pp:         out_channel -> t -> unit
    (* lattice values *)
    val init_t:     t (* initial abstract state *)
    val top:        t
    val is_bot:     t -> bool (* when returns true, t is bottom *)
    (* lattice operations *)
    val join:       t -> t -> t
    val widen:      expr list -> t -> t -> t
    val leq:        t -> t -> bool
    (* post-conditions for abstract commands *)
    val eval:       acmd -> t -> t
    (* computes a post-condition after entering and exiting a single
     * with clause *)
    val enter_with: withitem -> t -> t
    val exit_with:  withitem -> t -> t
    (* tries to prove that a condition holds
     * soundness: if sat e t returns true, all states in gamma(t) satisfy e *)
    val sat:        expr -> t -> bool
    (* checks whether a domain-specific relationship holds betweeen
     * two analysis results *)
    val is_related: t -> t -> bool
    (* compute the range info of a variable, given its variable name.
     * returns:
     *   None denotes unknown.
     *   Some(l,u,s) denotes list(range(l,u,s)) in python. *)
    val range_info: expr -> t -> (int * int * int) option
  end

(** Domain without bottom, and with externally managed dimensions *)
(* A bit experimental at the moment; useful under the fibered domain functor *)
module type ABST_DOMAIN_NB_D =
  sig
    include ABST_DOMAIN_NB
    (* Dimensions management *)
    val dim_add:         string -> t -> t (* add unexisting dimension *)
    val dim_rem:         string -> t -> t (* remove existing dimension *)
    val dim_project_out: string -> t -> t (* project info on dimension *)
    val dim_mem:         string -> t -> bool (* check if dimension exists *)
    val dims_get:        t -> SS.t option (* get all dimensions *)
    (* ad-hoc functions used by only some modules *)
    val set_aux_distty:  vtyp option -> t -> t (* used by adom_distty *)
    val bound_var_apron: string -> t -> int*int (* used by adom_apron *)
  end

(** Domain for numerical constraints + zone constraints *)
module type ABST_DOMAIN_ZONE_NB =
  sig
    include ABST_DOMAIN_NB_D
    (* Zone operations:
     *  zone_new_tensor   makes a tensor with empty zone
     *  zone_add_cell     augments the zone attached to a tensor by one cell
     *  zone_sat          checks equality of a zone with a hypercube *)
    val zone_dim_add: string -> t -> t
    val zone_dim_rem: string -> t -> t
    val zone_add_cell:   string -> bnd_expr list -> t -> t
      
    val zone_sat:        string -> (bnd_expr * bnd_expr) list -> t -> bool
    val zone_include:    bnd_expr list -> string -> t -> bool
      
    val zone_is_related: string -> t -> t -> bool
  end

(** Domain with bottom *)
module type ABST_DOMAIN_B =
  sig
    include ABST_DOMAIN_NB
    val bot:   t
  end

(** Signature for an Apron abstract domain manager wrapper
 **  (this is used to select Apron domain implementations) *)
module type APRON_MGR =
  sig
    val module_name: string
    type t
    val man: t Apron.Manager.t
  end
