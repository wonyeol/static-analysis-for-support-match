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
 ** ai_sig.ml: signature for analyzers *)
open Adom_sig
open Ir_sig

(* An analysis provides a main function computing an abstract post-condition *)
module type ANALYSIS =
  sig
    module Ad: ABST_DOMAIN_B
    val analysis_name: string
    val eval_prog: prog -> Ad.t -> (Ad.t * Ad.t * Ad.t)
    val analyze: prog -> (Ad.t * Ad.t * Ad.t)
    val pp: out_channel -> Ad.t -> unit
  end

(* An analyzer is parameterized by a state abstract domain *)
module type ANALYZER =
  functor (Ad: ABST_DOMAIN_B) -> (ANALYSIS with module Ad = Ad)
