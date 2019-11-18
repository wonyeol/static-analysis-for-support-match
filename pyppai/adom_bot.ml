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
 ** adom_bot.ml: abstract domain with bottom
 **)
open Lib
open Ir_sig
open Adom_sig

module Make = functor (D: ABST_DOMAIN_NB) ->
  (struct
    type t =
      | Bot
      | Nb of D.t

    let module_name = Printf.sprintf "+Bot(%s)" D.module_name

    let buf_t (buf: Buffer.t) (t: t): unit =
      match t with
      | Bot -> Printf.bprintf buf "_|_"
      | Nb u -> D.buf_t buf u
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    let bot = Bot
    let top = Nb D.top
    let is_bot = function
      | Bot -> true
      | Nb u -> D.is_bot u
    let init_t = Nb D.init_t

    let lift f = function
      | Bot -> Bot
      | Nb u -> try Nb (f u) with Bottom -> Bot

    let eval ac = lift (D.eval ac)
    let enter_with withitem = lift (D.enter_with withitem)
    let exit_with withitem = lift (D.exit_with withitem)

    let sat (e: expr): t -> bool = function
      | Bot -> true
      | Nb u -> D.sat e u

    let bin_combine f (t0: t) (t1: t): t =
      match t0, t1 with
      | Bot, t | t, Bot -> t
      | Nb u0, Nb u1 -> Nb (f u0 u1)
    let join = bin_combine D.join
    let widen thr = bin_combine (D.widen thr)

    let leq (t0: t) (t1: t): bool =
      match t0, t1 with
      | Bot, _ -> true
      | Nb u0, Bot -> D.is_bot u0
      | Nb u0, Nb u1 -> D.leq u0 u1

    let is_related t1 t2 =
      match t1, t2 with
      | Bot, Bot -> true
      | Bot, _ | _, Bot -> false
      | Nb nt1, Nb nt2 -> D.is_related nt1 nt2

    let range_info e = function
      | Bot -> None
      | Nb u -> D.range_info e u
  end: ABST_DOMAIN_B)
