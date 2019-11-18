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
 ** ir_cast.ml: conversion from Pyast_sig to Ir_sig *)
open Ir_sig
open Lib

module Pya = Pyast_sig
module IU = Ir_util
module PU = Pyast_util


(** ***************************)
(** Conversion of basic types *)
(** ***************************)
(** to_{idtf, number, uop, bop_{bool,other}, cop, dist_kind} *)
let to_idtf (id: Pya.identifier): idtf = id

let to_number: Pya.number -> number = function
  | Pya.Int   (i) -> Int   (i)
  | Pya.Float (f) -> Float (f)

let to_uop: Pya.unaryop -> uop = function
  | Pya.Not -> Not
  | Pya.Invert | Pya.UAdd | Pya.USub ->
      failwith "to_uop: unimplemented case"

let to_bop_unary: Pya.unaryop -> bop = function
  | Pya.UAdd -> Add
  | Pya.USub -> Sub
  | Pya.Not | Pya.Invert ->
      failwith "to_bop_unary: unimplemented case"

let to_bop_bool: Pya.boolop -> bop = function
  | Pya.And -> And
  | Pya.Or  -> Or

let to_bop_other: Pya.operator -> bop = function
  | Pya.Add  -> Add
  | Pya.Sub  -> Sub
  | Pya.Mult -> Mult
  | Pya.Div  -> Div
  | Pya.Pow  -> Pow
  | Pya.MatMult | Pya.Mod
  | Pya.LShift | Pya.RShift | Pya.BitOr
  | Pya.BitXor | Pya.BitAnd | Pya.FloorDiv ->
      failwith "to_bop_other: unimplemented case"

let to_cop: Pya.cmpop -> cop = function
  | Pya.Eq    -> Eq
  | Pya.NotEq -> NotEq
  | Pya.Lt    -> Lt
  | Pya.LtE   -> LtE
  | Pya.Gt    -> Gt
  | Pya.GtE   -> GtE
  | Pya.Is    -> Is
  | Pya.IsNot -> NotIs
  | Pya.In | Pya.NotIn ->
      failwith "to_cop: unimplemented case"

let dist_kind_list: string list =
  [ (*  0 *) "Normal";
    (*  1 *) "Exponential";
    (*  2 *) "Gamma";
    (*  3 *) "Beta";
    (*  4 *) "Uniform";
    (*  5 *) "Dirichlet";
    (*  6 *) "Poisson";
    (*  7 *) "Categorical";
    (*  8 *) "Bernoulli";
    (*  9 *) "OneHotCategorical";
    (* 10 *) "Delta" ]

let to_dist_kind (dist_kind_str: string): dist_kind =
  match list_find_index dist_kind_str dist_kind_list with
  |  0 -> Normal
  |  1 -> Exponential
  |  2 -> Gamma
  |  3 -> Beta
  |  4 -> Uniform None
  |  5 -> Dirichlet None
  |  6 -> Poisson
  |  7 -> Categorical None
  |  8 -> Bernoulli
  |  9 -> OneHotCategorical None
  | 10 -> Delta
  | (-1) -> failwith "to_dist_kind: unimplemented case"
  |    _ -> failwith "to_dist_kind: unreachable"


(** *************************************************)
(** Conversion of main language components: to_expr *)
(** *************************************************)
(** to_{expr{,_list}, dist_expr, keyword{,_list}} *)
(* helper functions *)
(* pyro-specific pattern matching for function calls *)
let _check field (e: 'a Pya.expr): bool * string =
  match field with
  | `Sample ->
     (match e with
      | Pya.Attribute(Pya.Name("pyro", _, _), "sample", _, _) ->
         (true, "")
      | _ ->
         (false, ""))
  | `StrFormat ->
     (match e with
      | Pya.Attribute(Pya.Str(s, _), "format", _, _) ->
         (true, s)
      | _ ->
         (false, ""))
let check      field (e: 'a Pya.expr): bool = fst (_check field e)
let check_info field (e: 'a Pya.expr): string = snd (_check field e)

let get_name: string option -> string = function
  | None -> NameManager.create_name ()
  | Some(name) -> name

(* to_expr *)
let rec to_expr ?(target: string option = None) :
          'a Pya.expr -> acmd list * expr = function
  | Pya.BoolOp (bop, [e1;e2], _) ->
      let new_bop = to_bop_bool bop in
      let (acmd_list1, new_e1) = to_expr e1 in
      let (acmd_list2, new_e2) = to_expr e2 in
      (acmd_list1 @ acmd_list2, BOp(new_bop, new_e1, new_e2))
  | Pya.BoolOp _ ->
      failwith "to_expr: unimplemented case (BoolOp)"
  | Pya.BinOp (e1, bop, e2, _) ->
      let new_bop = to_bop_other bop in
      let (acmd_list1, new_e1) = to_expr e1 in
      let (acmd_list2, new_e2) = to_expr e2 in
      (acmd_list1 @ acmd_list2, BOp(new_bop, new_e1, new_e2))
  | Pya.UnaryOp (uop, e, _) ->
      let (acmd_list, new_e) = to_expr e in
      begin
        match uop with
        | Pya.UAdd | Pya.USub ->
           let new_bop = to_bop_unary uop in
           (acmd_list, BOp(new_bop, Num(Int(0)), new_e))
        | Pya.Not | Pya.Invert ->
           let new_uop = to_uop uop in
           (acmd_list, UOp(new_uop, new_e))
      end
  | Pya.Dict (keys, vals, _) ->
      let (acmd_list1, new_keys) = to_expr_list keys in
      let (acmd_list2, new_vals) = to_expr_list vals in
      (acmd_list1 @ acmd_list2, Dict(new_keys, new_vals))
  | Pya.Compare (e1, [cop], [e2], _) ->
      let new_cop = to_cop cop in
      let (acmd_list1, new_e1) = to_expr e1 in
      let (acmd_list2, new_e2) = to_expr e2 in
      (acmd_list1 @ acmd_list2, Comp(new_cop, new_e1, new_e2))
  | Pya.Compare _ ->
      failwith "to_expr: unimplemented case (Compare)"
  | Pya.Num (n, _) ->
      ([], Num(to_number n))
  | Pya.Str (s, _) ->
      ([], Str(s))
  | Pya.NameConstant (None, _) ->
      ([], Nil)
  | Pya.NameConstant (Some true, _) ->
      ([], True)
  | Pya.NameConstant (Some false, _) ->
      ([], False)
  | Pya.Name(id, _, _) ->
      ([], Name(to_idtf id))

  | Pya.Call (f, [rv_name; dist_expr], kwargs, _) when check `Sample f ->
      let (acmd_list1, new_rv_name) = to_expr rv_name in
      let (acmd_list2, new_dist_name, new_dist_args) =
        to_dist_expr [] dist_expr in
      begin
        match kwargs with
        | [] | [(Some "infer", _)] ->
            (* Sample: acmd *)
            let new_var = get_name target in
            let new_acmd =
              Sample(new_var, new_rv_name, new_dist_name,
                     new_dist_args, None) in
            (acmd_list1 @ acmd_list2 @ [new_acmd], Name(new_var))
        | [(Some "obs", obs)] ->
            (* Sample or observe: acmd *)
            let (acmd_list3, new_obs) = to_expr obs in
            let new_var = get_name None in
            let new_acmd =
              Sample(new_var, new_rv_name, new_dist_name,
                      new_dist_args, Some new_obs) in
            (acmd_list1 @ acmd_list2 @ acmd_list3 @ [new_acmd], Name(new_var))
        | _ ->
            failwith "to_expr: unimplemented case (Call: pyro.sample: other)"
      end
  | Pya.Call (f, args, [], _) when check `StrFormat f ->
      (* StrFmt: expr *)
      let s = check_info `StrFormat f in
      let (acmd_list, new_args) = to_expr_list args in
      (acmd_list, StrFmt(s, new_args))
  | Pya.Call (f, args, kwargs, _) ->
      (* AssnCall: acmd *)
      let new_f =
        match f with
        | Pya.Name(f_name, _, _) ->
            Name (f_name)
        | Pya.Attribute(Pya.Name(obj_name, _, _), method_name, _, _) ->
            Name (obj_name ^ "." ^ method_name)
        | Pya.Attribute(Pya.Attribute(Pya.Name(obj_name, _, _),
              method_name1, _, _), method_name2, _, _) ->
            Name (obj_name ^ "." ^ method_name1 ^ "." ^ method_name2)
        | Pya.Attribute(Pya.Attribute(Pya.Attribute(Pya.Name(obj_name, _, _),
              method_name1, _, _), method_name2, _, _), method_name3, _, _) ->
            Name (obj_name ^ "." ^ method_name1 ^ "." ^ method_name2 ^ "."
                  ^ method_name3)
        | _ ->
            failwith "to_expr: unimplemented case (Call: other 1)"
      in
      let (acmd_list1, new_args) = to_expr_list args in
      let (acmd_list2, new_kwargs) = to_keyword_list kwargs in
      let new_var = get_name target in
      let new_acmd = AssnCall(new_var, new_f, new_args, new_kwargs) in
      (* special handling for `pyro.plate(...)' *)
      let acmd_list3 =
        match new_f with
        | Name "pyro.plate" ->
           (* set `new_var._indices' *)
           let acmd1 = 
             match new_args with
             | ((Str _) as arg0) :: args1 ->
                let x = NameManager.plate_indices_name new_var in
                if args1 = [] then
                  (* if no `size' is given to the plate, do:
                   * `__@@_indices(new_var) = None'. *)
                  (* FACT: if p is a plate with no `size', then `p._indices = None'. *)
                  Assn (x, Nil)
                else
                  (* otherwise, do:
                   * `__@@_indices(new_var) = Sample(name, Subsample(None, None), args1)' *)
                  let dist = (Subsample (None, None), []) in
                  Sample (x, arg0, dist, args1, None)
             | _ ->
                failwith "to_expr: unimplemented case (Call: other 2)" in
           (* (* set `new_var.dim' *)
           let acmd2 =
             (* do `__@dim(new_var) = None', if `dim' is not given in args1.
              * do `__@dim(new_var) = dim_n', if args1 = [...; (Some 'dim', dim_n); ...]. *)
             let x = NameManager.plate_dim_name new_var in
             let f (acc:expr) ((kwarg_name, kwarg_val): idtf option * expr): expr =
               if kwarg_name = Some "dim" then kwarg_val else acc in
             let dim_n = List.fold_left f Nil new_kwargs in
             Assn (x, dim_n) in *)
           [acmd1; (*acmd2*)]
        | _ -> [] in
      (acmd_list1 @ acmd_list2 @ [new_acmd] @ acmd_list3, Name(new_var))

  | Pya.Attribute (e1, attr, _, _) ->
      (* hy: our translated attribute is very approximate. It loses all
       * the information in the original expression *)
      let (acmd_list1, new_e1) = to_expr e1 in
      let new_var = get_name target in
      let new_acmd = AssnCall(new_var, Name("RYLY"), [], []) in
      (acmd_list1 @ [new_acmd], Name(new_var))
  | Pya.Subscript (e1, e2, _, _) ->
      (* AssnCall: acmd -- "access_with_index" *)
      let (acmd_list1, new_e1) = to_expr e1 in
      let (acmd_list2, new_e2_list) = to_slice_expr e2 in
      let new_var = get_name target in
      let new_acmd =
        AssnCall (new_var,
                  Name("access_with_index"), new_e1 :: new_e2_list, []) in
      (acmd_list1 @ acmd_list2 @ [new_acmd], Name(new_var))
  | Pya.List (elts, _, _)
  | Pya.Tuple (elts, _, _) ->
      let (acmd_list, new_elts) = to_expr_list elts in
      (acmd_list, List(new_elts))

and to_dist_expr
    (dist_trans_l: dist_trans list) :
      'a Pya.expr -> acmd list * dist * expr list = function
  | Pya.Call(Pya.Name(dist_kind, _, _), dist_args, _ (* ignore kwargs *), _) ->
      let new_dist_kind = to_dist_kind dist_kind in
      let new_dist = (new_dist_kind, dist_trans_l) in
      let (acmd_list, new_dist_args) = to_expr_list dist_args in
      (acmd_list, new_dist, new_dist_args)

  | Pya.Call (Pya.Attribute(dist_expr, attr, _, _), [s], _, _)
    when attr = "to_event" ->
      let (acmd_list1, new_s) = to_expr s in
      let new_dist_trans_l =
        match new_s with
        | Num(Int n) -> ToEvent(Some n) :: dist_trans_l
        | _          -> ToEvent(None)   :: dist_trans_l in
      let (acmd_list2, new_dist, new_dist_args) =
        to_dist_expr new_dist_trans_l dist_expr in
      (acmd_list1 @ acmd_list2, new_dist, new_dist_args)

  | Pya.Call (Pya.Attribute(dist_expr, attr, _, _), [s], _, _)
    when attr = "expand_by" ->
      let (acmd_list1, new_s) = to_expr s in
      let new_dist_trans_l =
        match new_s with
        | List(size) ->
            let new_dist_size = List.map IU.expr_to_int_opt size in
            ExpandBy(new_dist_size) :: dist_trans_l
        | _ ->
            failwith "to_dist_expr: unimplemented case 1" in
      let (acmd_list2, new_dist, new_dist_args) =
        to_dist_expr new_dist_trans_l dist_expr in
      (acmd_list1 @ acmd_list2, new_dist, new_dist_args)

  | Pya.Call (Pya.Attribute(dist_expr, attr, _, _), [s], _, _)
    when attr = "mask" ->
      to_dist_expr dist_trans_l dist_expr
  | dist_expr ->
      failwith "to_dist_expr: unimplemented case 2"

and to_slice_expr
    : 'a Pya.slice -> acmd list * expr list = function
  | Pya.Index (e) ->
     let (acmd_list, new_e) = to_expr e in
     (acmd_list, [new_e])
  | Pya.Slice (eopt_l, eopt_u, eopt_s) ->
     let (acmd_list_l, new_eopt_l) = to_expr_opt eopt_l in
     let (acmd_list_u, new_eopt_u) = to_expr_opt eopt_u in
     let (acmd_list_s, new_eopt_s) = to_expr_opt eopt_s in
     (* wy: below, Nil represents (the end of an array (or list))+1. *)
     let new_e_l = opt_get_default (Num(Int( 0))) new_eopt_l in
     let new_e_u = opt_get_default (Nil         ) new_eopt_u in
     let new_e_s = opt_get_default (Num(Int( 1))) new_eopt_s in
     (acmd_list_l @ acmd_list_u @ acmd_list_s,
      [List([new_e_l; new_e_u; new_e_s])])
  | Pya.ExtSlice (slice_list) ->
     let (acmd_list, new_e_list) =
       lift_to_list to_slice_expr slice_list in
     (acmd_list, List.concat new_e_list)

and to_expr_opt
    : 'a Pya.expr option -> acmd list * expr option = function
  | None ->
      ([], None)
  | Some e ->
      let (acmd_list, new_e) = to_expr e in
      (acmd_list, Some new_e)

and to_expr_list
      (e_list: 'a Pya.expr list): acmd list * expr list =
  lift_to_list to_expr e_list

and to_keyword
      ((id_opt, e): 'a Pya.keyword): acmd list * (idtf option * expr) =
  let (acmd_list, new_e) = to_expr e in
  match id_opt with
  | None ->
      (acmd_list, (None, new_e)) (* wy: does this case compile in python? *)
  | Some(id) ->
      (acmd_list, (Some(to_idtf id), new_e))

and to_keyword_list
      (k_list: 'a Pya.keyword list): acmd list * keyword list =
  lift_to_list to_keyword k_list


(** *************************************************)
(** Conversion of main language components: to_misc *)
(** *************************************************)
(** to_withitem{,_list} *)
let to_withitem ((context, e_opt): 'a Pya.withitem)
    : acmd list * withitem =
  let (acmd_list1, new_context) = to_expr context in
  let (acmd_list2, new_e_opt1) = to_expr_opt e_opt in
  (acmd_list1 @ acmd_list2, (new_context, new_e_opt1))

let to_withitem_list
      (l: 'a Pya.withitem list): acmd list * withitem list =
  lift_to_list to_withitem l


(** ******************************************************)
(** Conversion of main language components: stmt_to_prog *)
(** ******************************************************)
(* helper functions *)
let skip_prog: stmt = Atomic(Assume(True))

let check_assumption_on_stmt (stmt: 'a Pya.stmt): bool =
  not (PU.contains_middle_return stmt)

let append_prog_list (p_list: stmt list) (p_base: stmt list): stmt list =
  match p_base, List.rev p_list with
  | [ Atomic (Assume True) ], [p_last] ->
      [p_last]
  | [ Atomic (Assume True) ], p_last::p_rest ->
      List.fold_left (fun acc p0 -> p0 :: acc) [ p_last ] p_rest
  | _, p_list_rev ->
      List.fold_left (fun acc p0 -> p0 :: acc) p_base p_list_rev

let append_acmd_list (a_list: acmd list) (p_base: stmt list): stmt list =
  let p_list = List.map (fun a -> Atomic(a)) a_list in
  append_prog_list p_list p_base

(* let get_upper_bound_from_range: 'a Pya.expr -> 'a Pya.expr option = function
  | Pya.Call(f, args, _, _) ->
      begin
        match f, args with
        | Pya.Name("range", _, _), [upper_bound] ->
            Some upper_bound
        | Pya.Attribute(Pya.Name("pyro", _, _), attr, _, _), _
            :: upper_bound :: _  ->
            if List.mem attr ["irange"; "plate"] then
              Some upper_bound
            else
              None
        | _ ->
            None
      end
  | _ ->
      None *)

(* _stmt{,_list}_to_prog *)
let rec _stmt_to_prog
    (final_prog_list: stmt list)
    : 'a Pya.stmt -> stmt list = function
  | Pya.FunctionDef _ ->
      failwith "_stmt_to_prog: unimplemented case (FunctionDef)"
  | Pya.Return (None, _) ->
      [ skip_prog ]
  | Pya.Return (Some(value), _) ->
      let return_var_name = NameManager.return_name in
      let (acmd_list, new_value) =
        to_expr ~target:(Some(return_var_name)) value in
      if (new_value = Name(return_var_name)) then
        append_acmd_list acmd_list [ skip_prog ]
      else
        let new_return = Atomic(Assn(return_var_name, new_value)) in
        append_acmd_list acmd_list [ new_return ]
  | Pya.Assign ([Pya.Name(target_id, _, _)], value, _) ->
      let (acmd_list, new_value) =
        to_expr ~target:(Some(target_id)) value in
      if (new_value = Name(target_id)) then
        append_acmd_list acmd_list [ skip_prog ]
      else
        let new_assign = Atomic(Assn(target_id, new_value)) in
        append_acmd_list acmd_list [ new_assign ]
  | Pya.Assign ([Pya.Tuple(es, _, _)], value, _)
  | Pya.Assign ([Pya.List (es, _, _)], value, _) ->
      let (acmd_list, new_value) = to_expr value in
      let f (acmdl0, i) e0 =
        match e0 with
        | Pya.Name(target_id0, _, _) ->
            let acmd = AssnCall(target_id0, Name("access_with_index"),
                                [new_value; Num(Int(i))], []) in
            (acmd :: acmdl0, i+1)
        | _ -> failwith "_stmt_to_prog: unimplemented case (Assign, tuple)" in
      let (new_assign_list, _) = List.fold_left f ([], 0) es in
      let new_assign_list = List.rev new_assign_list in
      append_acmd_list (acmd_list @ new_assign_list) [ skip_prog ]
  | Pya.Assign ([Pya.Subscript(Pya.Name(target_id, _, _), slice, _, _)], value, _) ->
      (* target_id[slice] = value *)
      let (acmd_list1, new_slice) = to_slice_expr slice in
      let (acmd_list2, new_value) = to_expr value in
      let new_assign = Atomic(AssnCall(target_id, Name("update_with_index"),
                                       [Name(target_id); List(new_slice); new_value], [])) in
      append_acmd_list (acmd_list1 @ acmd_list2) [ new_assign ]
  | Pya.Assign ([              Pya.Attribute(Pya.Name(target_id, _, _), _, _, _)          ],
                value, _)
  | Pya.Assign ([Pya.Attribute(Pya.Attribute(Pya.Name(target_id, _, _), _, _, _), _, _, _)],
                value, _) ->
      (* target_id.*   = value, or
       * target_id.*.* = value *)
      let (acmd_list, new_value) = to_expr value in
      let new_assign = Atomic(AssnCall(target_id, Name("update_with_field"),
                                       [Name(target_id); Nil; new_value], [])) in
      append_acmd_list acmd_list [ new_assign ]
  | Pya.Assign _ ->
      failwith "_stmt_to_prog: unimplemented case (Assign)"
  | Pya.For(trgt, iter, body, [], _) ->
     let (acmd_list_iter, new_iter) = to_expr iter in
     let (acmd_list_trgt, new_trgt) = to_expr trgt in
     let new_body = _stmt_list_to_prog final_prog_list body in
     let new_for = For(new_trgt, new_iter, new_body) in
     append_acmd_list (acmd_list_iter @ acmd_list_trgt) [ new_for ]
  (* (* wy: this part is moved to ai_make. *)
  | Pya.For (Pya.Name(i, _, _), range_expr, body, [], _) ->
      begin
        (* range_expr --> (acmd_list_init, _) *)
        let (acmd_list_init, _) = to_expr range_expr in
        match get_upper_bound_from_range range_expr with
        | None ->
            failwith "_stmt_to_prog: unimplemented case (For)"
        | Some(upper_bound) ->
            (* upper_bound --> (acmd_list, new_upper_bound) *)
            let (acmd_list, new_upper_bound) = to_expr upper_bound in
            let var_i = Name(i) in
            (* initialize_i= i=0 *)
            (* increase_i  = i=i+1 *)
            (* test_i      = i<new_upper_bound *)
            let initialize_i = Atomic(Assn(i, Num(Int(0)))) in
            let increase_i = Assn(i, BOp(Add, var_i, Num(Int(1)))) in
            let test_i = Comp(Lt, var_i, new_upper_bound) in
            (* initialize= acmd_list_init; i=0 *** *)
            (* eplilogue = i=i+1; acmd_list *)
            (* new_body  = body *)
            (* new_while = while(i<new_upper_bound){ body; i=i+1; acmd_list } *** *)
            let initialize = append_acmd_list acmd_list_init [ initialize_i ] in
            let epilogue: stmt list =
              append_acmd_list (increase_i::acmd_list) [ skip_prog ] in
            let new_body = _stmt_list_to_prog epilogue body in
            let new_while =
              append_acmd_list acmd_list
                [ While (test_i, new_body @ epilogue) ] in
            if (List.exists PU.contains_continue body
              || List.exists PU.contains_break body) then
              initialize @ new_while
            else
              (* assert_i      = acmd_list; assert(i<new_upper_bound) *** *)
              (* unrolled_while= body; i=i+1; new_while *** *)
              let assert_i =
                append_acmd_list acmd_list [ Atomic (Assert test_i) ] in
              let unrolled_while =
                new_body @ [ Atomic increase_i ] @ new_while in
              initialize @ assert_i @ unrolled_while
      end
   *)
  | Pya.For _ ->
      failwith "_stmt_to_prog: unimplemented case (For)"
  | Pya.While (test, body, [], _) ->
      let (acmd_list, new_test) = to_expr test in
      let prog_list = List.map (fun a -> Atomic(a)) acmd_list in
      let new_body1 = _stmt_list_to_prog prog_list body in
      let new_body2 = append_prog_list (new_body1 @ prog_list) [ skip_prog ] in
      let new_while =
        append_prog_list prog_list [ While (new_test, new_body2) ] in
      if (List.exists PU.contains_continue body
        || List.exists PU.contains_break body) then
        new_while
      else
        let assert_test =
          append_acmd_list acmd_list [ Atomic (Assert new_test) ] in
        assert_test @ new_body1 @ new_while
  | Pya.While _ ->
      failwith "_stmt_to_prog: unimplemented case (While)"
  | Pya.If (test, body, orelse, _) ->
      let (acmd_list, new_test) = to_expr test in
      let new_body =
        _stmt_list_to_prog final_prog_list body in
      let new_orelse =
        _stmt_list_to_prog final_prog_list orelse in
      let new_if = If (new_test, new_body, new_orelse) in
      append_acmd_list acmd_list [ new_if ]
  | Pya.With (items, body, _) ->
      let (acmd_list, new_items) =
        to_withitem_list items in
      let new_body =
        _stmt_list_to_prog final_prog_list body in
      let new_with = With(new_items, new_body) in
      append_acmd_list acmd_list [ new_with ]
  | Pya.Expr (e, _) ->
      let (acmd_list, new_e) = to_expr e in
      append_acmd_list acmd_list [ skip_prog ]
  | Pya.Pass _ ->
      [ skip_prog ]
  | Pya.Break _ ->
      [ Break ]
  | Pya.Continue _ ->
      append_prog_list final_prog_list [ Continue ]

and _stmt_list_to_prog
    (final_prog_list: stmt list)
    (stmt_list: 'a Pya.stmt list): prog =
  let prog_list =
    List.map (_stmt_to_prog final_prog_list) stmt_list in
  let prog_list_rev = List.rev prog_list in
  match prog_list_rev with
  | [] ->
      [ skip_prog ]
  | [ stmt ] ->
      stmt
  | prog_last::prog_rest ->
      List.fold_left (fun acc p -> p @ acc) prog_last prog_rest

(* stmt{,_list}_to_prog *)
let stmt_to_prog (stmt: 'a Pya.stmt): prog =
  if (check_assumption_on_stmt stmt) then
    _stmt_to_prog [] stmt
  else
    failwith "stmt_to_prog: assumption violation"

let stmt_list_to_prog (stmt_list: 'a Pya.stmt list): prog =
  if (List.for_all check_assumption_on_stmt stmt_list) then
    (* wy: I presume that `check_assumption_on_stmt' tries to check 
     *     whether `return' appears only at the end of a program. 
     *     However, I think the above check doesn't exclude the following case:
     *     `return 1; return 2'. Is it correct? Then needs to be fixed? *)
    _stmt_list_to_prog [] stmt_list
  else
    failwith "stmt_list_to_prog: assumption violation"


(** ******************************************************)
(** Conversion of main language components: modl_to_prog *)
(** ******************************************************)
let modl_to_prog: 'a Pya.modl -> prog = function
  | Pya.Module(stmt_list, _) -> stmt_list_to_prog stmt_list
