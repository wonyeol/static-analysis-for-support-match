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
 ** ai_make.ml: construction of a static analyzer *)
open Adom_sig
open Ai_sig
open Ir_sig

module IU = Ir_util

(** Global switch for iterator debugging information *)
let debug = ref false

(** Global switch for widening with thresholds *)
let wid_thr: bool ref = ref false

(** Global switch for simplifying expressions during the analysis of Assume *)
let sim_assm: bool ref = ref false


(** Generic functor to create analyzers *)
module MakeAnalysis : ANALYZER = functor (Ad: ABST_DOMAIN_B) ->
  struct
    let analysis_name = Ad.module_name
    module Ad = Ad

    let wfix (thr: expr list) f =
      let rec aux st =
        let st_new = f st in
        if Ad.leq st_new st then st
        else aux (Ad.widen thr st st_new) in
      aux

    let rec eval_stmt (p: stmt) st =
      if !debug then
        Printf.printf "\nAbout to analyze:\n%aAbstract state:\n%a\n\n"
          Ir_util.pp_stmt p Ad.pp st;
      match p with
      | Atomic ac ->
          Ad.eval ac st, Ad.bot, Ad.bot
      | If (e, p_t, p_f) ->
          let st_t, stc_t, stb_t =
            eval_block p_t (Ad.eval (Assume(e)) st) in
          let st_f, stc_f, stb_f =
            eval_block p_f (Ad.eval (Assume(UOp(Not,e))) st) in
          Ad.join st_t st_f, Ad.join stc_t stc_f, Ad.join stb_t stb_f
      | For (Name i as e_i, e_r, p_b) ->
          (* construct p_new dynamically *)
          let (cmp, e_l, e_u, e_s): cop * expr * expr * expr =
            match Ad.range_info e_r st with
            | None -> failwith "eval_stmt: for: unimplemented case 1"
            | Some (l, u, s) ->
               (if      s > 0 then Lt
                else if s < 0 then Gt
                else failwith "eval_stmt: for: error 1"),
               Num(Int l), Num(Int u), Num(Int s) in
          (* init_i= i=e_l *** FINAL
           * inc_i = i=i+e_s
           * test_i= i<e_u *)
          let init_i = Atomic(Assn(i, e_l)) in 
          let inc_i  = Atomic(Assn(i, BOp(Add, e_i, e_s))) in
          let test_i = Comp(cmp, e_i, e_u) in
          (* body_new     = body; i=i+e_s
           * while_new    = while (i<e_u){ body; i=i+e_s } *** FINAL
           * body_unrolled= assert(i<e_u); body; i=i+e_s   *** FINAL *)
          let body_new = p_b @ [inc_i] in
          let while_new = While(test_i, body_new) in
          let body_unrolled =
            if (List.exists IU.contains_continue p_b
                || List.exists IU.contains_break p_b)
            then (* do not unroll *)
              []
            else (* do unroll *)
              Atomic(Assert(test_i)) :: body_new in
          let p_new = [init_i] @ body_unrolled @ [while_new] in
          (* eval p_new *)
          eval_block p_new st
      | For _ -> failwith "eval_stmt: for: unimplemented case 2"
      | While (e, p_b) ->
          let f_iter st0 =
            let st1, st1c, _ =
              eval_block p_b (Ad.eval (Assume e) st0) in
            Ad.join st (Ad.join st1 st1c) in
          let st_fix =
            let thr = if !wid_thr then [ e ] else [ ] in
            wfix thr f_iter st in
          let st2, st2c, _ = eval_block p_b (Ad.eval (Assume e) st_fix)
              (* hy: one-step narrowing *) in
          let st_inv = Ad.join st (Ad.join st2 st2c) in
          let _, _, st_b = eval_block p_b st_inv in
          let st_filtered = Ad.eval (Assume(UOp(Not,e))) st_inv in
          Ad.join st_b st_filtered, Ad.bot, Ad.bot
      | With (item_l, p_b) ->
          let preprocess st0 =
            List.fold_left (fun st1 i -> Ad.enter_with i st1) st0 item_l in
          let postprocess st0 =
            List.fold_right Ad.exit_with item_l st0 in
          let st1 = preprocess st in
          let st2, st2_c, st2_b = eval_block p_b st1 in
          (postprocess st2, postprocess st2_c, postprocess st2_b)
      | Break ->
          (Ad.bot, Ad.bot, st)
      | Continue ->
          (Ad.bot, st, Ad.bot)
    and eval_block l st =
      match l with
      | [ ] -> st, Ad.bot, Ad.bot
      | [ p ] -> eval_stmt p st
      | p1 :: p2 ->
          let st1, st1c, st1b = eval_stmt p1 st in
          let st2, st2c, st2b = eval_block p2 st1 in
          (* XR: not sure about the joins here ! *)
          st2, Ad.join st1c st2c, Ad.join st1b st2b

    let eval_prog = eval_block
    let analyze p = eval_block p Ad.init_t
    let pp = Ad.pp
  end
