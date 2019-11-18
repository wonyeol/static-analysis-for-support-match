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
 ** irty_util.ml: utilities for types of ir *)
open Lib
open Ir_sig
open Irty_sig

exception Broadcast_failure
module IU = Ir_util

(** ***********)
(** to_string *)
(** ***********)
let num_ty_to_string = function
  | NT_int  -> "int"
  | NT_real -> "real"

let tensor_size_ty_to_string ts =
  match ts with
  | None -> "T"
  | Some l -> list_to_string int_opt_to_string ";" l

let plate_ty_to_string = function
  | None -> "T"
  | Some(size_args, named_args) ->
      let size_args_str = list_to_string int_opt_to_string ";" size_args in
      let named_args_str = smap_to_string int_opt_to_string ";" named_args in
      size_args_str ^ "," ^ named_args_str

let range_ty_to_string = function
  | None -> "T"
  | Some(args) ->
      list_to_string int_opt_to_string ";" args

let distr_ty_to_string = function
  | None -> "T"
  | Some(dk, tsl) ->
      Printf.sprintf "%s, %s"
        (IU.dist_kind_to_string dk)
        (list_to_string tensor_size_ty_to_string ";" tsl)

let fun_ty_to_string = function
  | FT_tens_resize(l1,l2,l3) -> 
      let s1 = list_to_string int_opt_to_string ";" l1 in
      let s2 = list_to_string int_opt_to_string ";" l2 in
      let s3 = list_to_string int_opt_to_string ";" l3 in
      "resize_t([" ^ s1 ^ "],[" ^ s2 ^ "],[" ^ s3 ^ "])"
  | FT_top -> "top"

let exp_ty_to_string = function
  | ET_bool      -> "bool"
  | ET_nil       -> "nil"
  | ET_num nt    -> num_ty_to_string nt
  | ET_plate pt  -> "plate[" ^ plate_ty_to_string pt ^ "]"
  | ET_range rt  -> "range[" ^ range_ty_to_string rt ^ "]"
  | ET_distr dt  -> "distr[" ^ distr_ty_to_string dt ^ "]"
  | ET_fun ft    -> "fun[" ^ fun_ty_to_string ft ^ "]"
  | ET_tensor ts -> "tens[" ^ tensor_size_ty_to_string ts ^"]"
  | ET_unknown   -> "unknown"

let vtyp_to_string = function
  | Vt_nil      -> "nil"
  | Vt_num nt   -> num_ty_to_string nt
  | Vt_plate pt -> "plate[" ^ plate_ty_to_string pt ^ "]"
  | Vt_range rt -> "range[" ^ range_ty_to_string rt ^ "]"
  | Vt_distr dt -> "distr[" ^ distr_ty_to_string dt ^ "]"
  | Vt_fun ft   -> "fun[" ^ fun_ty_to_string ft ^ "]"
  | Vt_tens ts  -> "tens[" ^ tensor_size_ty_to_string ts ^ "]"

let broadcast_info_to_string = function
  | [] -> "[]"
  | size_dim :: l ->
      let f (s0, d0) = "(" ^ string_of_int s0 ^ ", " ^ int_opt_to_string d0 ^ ")" in
      let size_dim_str = f size_dim in
      "[" ^ (List.fold_left (fun acc size_dim0 -> acc ^ ", " ^ (f size_dim0)) size_dim_str l) ^ "]"

let dist_size_ty_to_string (batch_size, event_size) =
  Printf.sprintf "(%s, %s)"
    (list_to_string int_opt_to_string ";" batch_size)
    (list_to_string int_opt_to_string ";" event_size)

(** *************************************)
(** LOCAL: leq and join for             *)
(** 'a option, 'a list, 'a StringMap.t  *)
(**                                     *)
(** - 'a of list & SM.t has leq & join  *)
(** - option is interpreted as top      *)
(** *************************************)
(* 'a option *)
let a_opt_discrete_leq (ao1: 'a option) (ao2: 'a option): bool =
  match ao1, ao2 with
  | _, None -> true
  | None, _ -> false
  | Some ao1, Some ao2 -> ao1 = ao2

let a_opt_discrete_join (ao1: 'a option) (ao2: 'a option): 'a option =
  match ao1, ao2 with
  | None, _ | _, None -> None
  | Some a1, Some a2 ->
     if a1 = a2 then ao1 else None

let int_opt_leq (io1: int option) (io2: int option): bool =
  a_opt_discrete_leq io1 io2

let int_opt_join (io1: int option) (io2: int option): int option =
  a_opt_discrete_join io1 io2

(* 'a list *)
let list_leq (leq: 'a -> 'a -> bool) (l1: 'a list) (l2: 'a list) : bool =
  if List.length l1 <> List.length l2 then false
  else List.for_all2 leq l1 l2

let list_join (join: 'a -> 'a -> 'a) (l1: 'a list) (l2: 'a list) : 'a list option =
  if List.length l1 <> List.length l2 then None
  else Some (List.map2 join l1 l2)

(* 'a SM.t *)
let smap_same_dom (m1: 'a SM.t) (m2: 'a SM.t): bool =
  SM.for_all (fun s1 _ -> SM.mem s1 m2) m1 
  && SM.for_all (fun s2 _ -> SM.mem s2 m1) m2 

let smap_leq (leq: 'a -> 'a -> bool) (m1: 'a SM.t) (m2: 'a SM.t) : bool =
  if not (smap_same_dom m1 m2) then false
  else SM.for_all (fun s1 v1 -> leq v1 (SM.find s1 m2)) m1

let smap_join (join: 'a -> 'a -> 'a) (m1: 'a SM.t) (m2: 'a SM.t) : 'a SM.t option =
  if not (smap_same_dom m1 m2) then None
  else Some (SM.union (fun _ v1 v2 -> Some (join v1 v2)) m1 m2)

(** *************************)
(** tensor_size_ty (GLOBAL) *)
(** *************************)
(* Check the order between tensor size types *)
let tensor_size_ty_leq ts1 ts2 =
  match ts1, ts2 with 
  | _, None -> true
  | None, _ -> false
  | Some(dl1), Some(dl2) -> list_leq int_opt_leq dl1 dl2

let tensor_size_ty_join ts1 ts2 =
  match ts1, ts2 with
  | None, _ | _, None -> None
  | Some(dl1), Some(dl2) -> list_join int_opt_join dl1 dl2

let tensor_size_ty_concat (ts1: tensor_size_ty) (ts2: tensor_size_ty): tensor_size_ty =
  match ts1, ts2 with
  | Some s1, Some s2 -> Some (s1 @ s2)
  | _ -> None

(** ********)
(** num_ty *)
(** ********)
(* Check the order between num types *)
let num_ty_leq nt1 nt2 =
  match nt1, nt2 with
  | NT_int, NT_int 
  | NT_int, NT_real -> true
  | NT_real, NT_int -> false
  | NT_real, NT_real -> true
                       
(* Join of num types, with subtyping *)
let num_ty_join ty1 ty2 =
  match ty1, ty2 with
  | NT_real, _ | _, NT_real -> NT_real
  | NT_int, NT_int -> NT_int

(** **********)
(** plate_ty *)
(** **********)
(* Check the order between plate types *)
let plate_ty_leq pt1 pt2 =
  match pt1, pt2 with
  | _, None -> true
  | None, _ -> false
  | Some (s_args1, n_args1), Some (s_args2, n_args2) ->
      list_leq int_opt_leq s_args1 s_args2 
      && smap_leq int_opt_leq n_args1 n_args2

(* Join of two plate types *)
let plate_ty_join pt1 pt2 =
  match pt1, pt2 with
  | None, _ | _, None -> None
  | Some (s_args1, n_args1), Some (s_args2, n_args2) ->
      begin
        let s_args_opt = list_join int_opt_join s_args1 s_args2 in
        let n_args_opt = smap_join int_opt_join n_args1 n_args2 in
        match s_args_opt, n_args_opt with
        | None, _ | _, None -> None
        | Some s_args, Some n_args -> Some (s_args, n_args)
      end

(** **********)
(** range_ty *)
(** **********)
(* Check the order between range types *)
let range_ty_leq rt1 rt2 =
  match rt1, rt2 with
  | _, None -> true
  | None, _ -> false
  | Some (args1), Some (args2) ->
      list_leq int_opt_leq args1 args2 

(* Join of two range types *)
let range_ty_join rt1 rt2 =
  match rt1, rt2 with
  | None, _ | _, None -> None
  | Some (args1), Some (args2) ->
      list_join int_opt_join args1 args2

(** **********)
(** distr_ty *)
(** **********)
(* Check the order between distr types *)
let distr_ty_leq dt1 dt2 =
  match dt1, dt2 with
  | _, None -> true
  | None, _ -> false
  | Some (dk1, tsl1), Some (dk2, tsl2) ->
      (dk1 = dk2) &&
        (list_leq tensor_size_ty_leq tsl1 tsl2)

(* Join of two distr types *)
let distr_ty_join dt1 dt2 =
  match dt1, dt2 with
  | None, _ | _, None -> None
  | Some (dk1, tsl1), Some (dk2, tsl2) ->
      begin
        let tsl_opt = list_join tensor_size_ty_join tsl1 tsl2 in
        match (dk1=dk2), tsl_opt with
        | false, _ | _, None -> None
        | true, Some tsl -> Some (dk1, tsl)
      end

(** ********)
(** fun_ty *)
(** ********)
(* Check the order between fun types *)
let fun_ty_leq ft1 ft2 =
  match ft1, ft2 with 
  | _, FT_top -> true
  | FT_top, _ -> false
  | FT_tens_resize(i1,o1,n1), FT_tens_resize(i2,o2,n2) ->
      (list_leq int_opt_leq i1 i2) 
      && (list_leq int_opt_leq o1 o2)
      && (list_leq int_opt_leq n1 n2)

(* Join of fun types, with subtyping *)
let fun_ty_join ft1 ft2 =
  match ft1, ft2 with
  | _, FT_top | FT_top, _ -> FT_top
  | FT_tens_resize(i1,o1,n1), FT_tens_resize(i2,o2,n2) ->
      let i_opt = list_join int_opt_join i1 i2 in
      let o_opt = list_join int_opt_join o1 o2 in
      let n_opt = list_join int_opt_join n1 n2 in
      begin
        match i_opt, o_opt, n_opt with
        | None, _, _ | _, None, _ | _, _, None -> FT_top
        | Some i, Some o, Some n -> FT_tens_resize(i,o,n)
      end

(* Apply a function type on a tensor size type *)
let fun_ty_apply (ft: fun_ty) (ts: tensor_size_ty) : tensor_size_ty =
  match ft with
  | FT_tens_resize(l1,l2,l3) ->
      begin
        let rec truncate (s0: int option list) (s1: int option list) =
          match s0, s1 with
          | _, [] -> 
              s0
          | [], _ -> 
              raise (Must_error "fun_ty_apply: tensor operator applied incorrectly")
          | _::r0, None::r1 | None::r0, _::r1 -> 
              truncate r0 r1 
          | (Some n0)::r0, (Some n1)::r1 -> 
              if n0 = n1 then truncate r0 r1
              else raise (Must_error "fun_ty_apply: tensor operator applied incorrectly")
        in
        match ts with
        | None -> None 
        | Some l0 -> 
            let l0_trunc = List.rev (truncate (List.rev l0) (List.rev l1)) in
            Some (l3 @ l0_trunc @ l2)
      end
  | _ -> None    

(** ************************)
(** tensor_size_ty (LOCAL) *)
(** ************************)
(* Function list: 
 * - tensor_size_ty_do_broadcast: tensor_size_ty list -> tensor_size_ty
 * - tensor_size_ty_from_broadcast_info: broadcast_info -> tensor_size_ty *)

(* Broadcasts two tensor sizes against each other, and returns the result.
 *
 * The function adopts our usual optimistic assumption about python
 * runtime check. It filters out executions where this check fails, 
 * and computes a result that conservatively describes outcomes
 * of unfiltered executions. The function raises the Broadcast_failure 
 * exception to signal the definite or must failure of the runtime check.
 *)
let _broadcast (ts1: tensor_size_ty) (ts2: tensor_size_ty) : tensor_size_ty =
  match ts1, ts2 with
  | None, _ | _, None -> None
  | Some t1, Some t2 ->
      let broadcast_size (s1: int option) (s2: int option): int option =
        match s1, s2 with
        | None, None ->
            None
        | None, Some(n2) ->
            if (n2=1) then None else s2
        | Some(n1), None ->
            if (n1=1) then None else s1
        | Some(n1), Some(n2) ->
            if (n1 = n2) then s1
            else if (n1 = 1) then s2
            else if (n2 = 1) then s1
            else raise Broadcast_failure in
      let rec broadcast_tensor (l1: int option list) (l2: int option list)
          : int option list =
        match l1, l2 with
        | [], _ -> l2
        | _, [] -> l1
        | s1::rest1, s2::rest2 ->
            (broadcast_size s1 s2)::(broadcast_tensor rest1 rest2)
      in
      Some (List.rev (broadcast_tensor (List.rev t1) (List.rev t2)))

let tensor_size_ty_do_broadcast (ts_lst: tensor_size_ty list): tensor_size_ty = 
  List.fold_left _broadcast (Some []) ts_lst

(* Create a tensor size type from broadcast information *)
let tensor_size_ty_from_broadcast_info (bc_info: broadcast_info): tensor_size_ty =
  (* bc_info = [(64,None); (32,None); (4,-4)] *)
  let f ((m, i): int IntMap.t * int) = function
    | (size, None) -> failwith "tensor_size_ty_from_broadcast_info: unreachable"
        (* let rec find_next_i (i_cur: int): int =
          if (IntMap.mem i_cur m)
          then find_next_i (i_cur-1)
          else i_cur in
        let i_new = find_next_i i in
        (IntMap.add i_new size m, i_new-1) *)
    | (size, Some j) ->
        if (IntMap.mem j m) then 
          raise (Must_error "tensor_size_ty_from_broadcast_info: >1 broadcast on one dim")
        else 
          (IntMap.add j size m, i) in
  let (bc_map, _): int IntMap.t * int = List.fold_left f (IntMap.empty,-1) bc_info in
  (* bc_map = {-2:32, -1:64, -4:4} *)
  let bc_bindings: (int*int) list =
    IntMap.bindings bc_map |> List.sort (fun (d1,_) (d2,_) -> d2-d1) in
  (* bc_bindings = [(-1,64); (-2,32); (-4,4)] *)
  let g ((l, i): int option list * int) ((j, size): int * int) =
    if (i = j) then 
      ((Some size) :: l, i - 1) 
    else if (j < i) then 
      ((Some size) :: (list_repeat (i-j) (Some 1)) @ l, j - 1)
    else
       failwith "tensor_size_ty_from_broadcast_info: violated internal invariant" in
  let (bc_pre_ts, _): int option list * int = (List.fold_left g ([],-1) bc_bindings) in
  (* bc_pre_ts = [Some 4, Some 1, Some 32, Some 64] *)
  Some bc_pre_ts

(** ********)
(** exp_ty *)
(** ********)
(* get_exp_ty tries to find partially-correct type information.
 * The "partial" refers to the fact that this function assumes
 * the success of python's runtime type checker. Thus, if the checker
 * raises a runtime exception for an expression, the function 
 * is allowed to return anything. *)
let get_exp_ty get_name_ty: expr -> exp_ty =
  let rec aux = function
    | Nil ->
        ET_nil
    | True | False ->
        ET_bool
    | UOp (Not, _) | UOp (SampledStr, _) | UOp(SampledStrFmt, _) ->
        ET_bool
    | Comp (_, e1, e2) ->
        let e1_ty = aux e1 in
        let e2_ty = aux e2 in 
        begin
          match e1_ty, e2_ty with
          | ET_unknown, _ | _, ET_unknown -> ET_unknown
          (* e1_ty = ET_tensor || e2_ty = ET_tensor *)
          | ET_tensor(ts1), ET_tensor(ts2) ->
             ET_tensor(tensor_size_ty_do_broadcast [ts1; ts2])
          | ET_tensor _, ET_num _ | ET_tensor _, ET_bool -> e1_ty
          | ET_num _, ET_tensor _ | ET_bool, ET_tensor _ -> e2_ty
          (* e1_ty != ET_tensor || e2_ty != ET_tensor *)
          | _, _ -> ET_bool
        end 
    | Num(Int _) ->
        ET_num(NT_int)
    | Num(Float _) ->
        ET_num(NT_real)
    | List _ | Dict _ | Str _ | StrFmt _ ->
        ET_unknown
    | Name x -> 
        get_name_ty x
    | BOp (bop, e1, e2) ->
        begin
          let e1_ty = aux e1 in
          let e2_ty = aux e2 in 
          match bop with
          | And | Or ->
              begin 
                match e1_ty, e2_ty with 
                (* https://docs.python.org/3/reference/expressions.html#boolean-operations *)
                | ET_bool, ET_bool -> ET_bool
                | ET_num(n1_ty), ET_num(n2_ty) ->
                    ET_num(num_ty_join n1_ty n2_ty)
                | _ -> ET_unknown
              end
         | Add | Sub | Mult | Div | Pow ->
             begin
               match e1_ty, e2_ty with
               | ET_nil, _ | _, ET_nil -> ET_unknown
               | ET_plate _, _ | _, ET_plate _ -> ET_unknown
               | ET_range _, _ | _, ET_range _ -> ET_unknown
               | ET_distr _, _ | _, ET_distr _ -> ET_unknown
               | ET_fun _, _ | _, ET_fun _ -> ET_unknown
               | ET_unknown, _ | _, ET_unknown -> ET_unknown
               (* e1_ty = ET_tensor || e2_ty = ET_tensor *)
               | ET_tensor(ts1), ET_tensor(ts2) ->
                  ET_tensor(tensor_size_ty_do_broadcast [ts1; ts2])
               | ET_tensor _, ET_bool | ET_tensor  _, ET_num _ -> e1_ty
               | ET_bool, ET_tensor _ | ET_num _, ET_tensor  _ -> e2_ty
               (* e1_ty != ET_tensor && e2_ty != ET_tensor *)
               | ET_bool, ET_bool -> ET_num(NT_int)
               | _, ET_bool -> e1_ty
               | ET_bool, _ -> e2_ty
               | ET_num(n1_ty), ET_num(n2_ty) ->
                  begin
                    match bop with
                    | Add | Sub | Mult -> ET_num(num_ty_join n1_ty n2_ty)
                    | Div | Pow -> ET_num(NT_real)
                    | _ -> failwith "unreachable"
                  end
             end
       end in
  aux

(* is_{int,real,tensor,fun}_exp *)
let is_int_exp get_name_ty e =
  match get_exp_ty get_name_ty e with
  | ET_num(NT_int) -> true
  | _ -> false
let is_real_exp get_name_ty e =
  match get_exp_ty get_name_ty e with
  | ET_num _ -> true
  | _ -> false
let is_tensor_exp get_name_ty e =
  match get_exp_ty get_name_ty e with
  | ET_tensor _ -> true
  | _ -> false
let is_fun_exp get_name_ty e =
  match get_exp_ty get_name_ty e with
  | ET_fun _ -> true
  | _ -> false

(* checks whether a given type can be lifted to a tensor *)
let is_broadcastable_exp_ty = function
  (* wy: Do we need to allow every type to be broadcastable?
   *     For now, at least Nil is set to be broadcastable.
   *     Doing so looks not so strange... *)
  | ET_bool | ET_num _ | ET_tensor _ -> true
  | ET_nil -> true
    (* e.g.: pyro.plate('p', 4, None) --> Subsample(4, None)
     *       So, None should be broadcastable. *)
  | ET_plate _ | ET_range _ | ET_distr _ | ET_fun _ | ET_unknown  -> false

(* broadcast a given list of expression types *)
let exp_ty_do_broadcast (ty_l: exp_ty list): tensor_size_ty = 
  let pick_tens_ty acc = function (ET_tensor ts) -> ts::acc | _ -> acc in
  let ty_l0 = List.rev (List.fold_left pick_tens_ty [] ty_l) in
  let ty_l1 = (Some []) :: ty_l0 in
  tensor_size_ty_do_broadcast ty_l1

(** ******)
(** vtyp *)
(** ******)
let vtyp_leq typ1 typ2 =
  match typ1, typ2 with
  | Vt_nil      , Vt_nil       -> true
  | Vt_nil      , _            -> false
  | Vt_num nt1  , Vt_num nt2   -> num_ty_leq nt1 nt2
  | Vt_num _    , _            -> false
  | Vt_plate pt1, Vt_plate pt2 -> plate_ty_leq pt1 pt2
  | Vt_plate _  , _            -> false
  | Vt_range rt1, Vt_range rt2 -> range_ty_leq rt1 rt2
  | Vt_range _  , _            -> false
  | Vt_distr dt1, Vt_distr dt2 -> distr_ty_leq dt1 dt2
  | Vt_distr _  , _            -> false
  | Vt_fun ft1  , Vt_fun ft2   -> fun_ty_leq ft1 ft2
  | Vt_fun _    , _            -> false
  | Vt_tens ts1 , Vt_tens ts2  -> tensor_size_ty_leq ts1 ts2
  | Vt_tens _   , _            -> false

(** ****************)
(** broadcast_info *)
(** ****************)
let next_dim_from_broadcast_info (bc_info: broadcast_info): int =
  let cur_dims_list = List.map (fun (_,dim_opt) -> opt_get_fail dim_opt) bc_info in
  let cur_dims_set = IntSet.of_list cur_dims_list in
  let rec find_next_i (i: int): int =
    if (IntSet.mem i cur_dims_set)
    then find_next_i (i-1)
    else i in
  find_next_i (-1)
             
(** **************)
(** dist_size_ty *)
(** **************)
module type DistSizeTy =
  sig
    val get_ty: exp_ty list -> dist_kind -> dist_size_ty option
    val apply_trans_l: dist_trans list -> dist_size_ty option -> dist_size_ty option
    val apply_broadcast: broadcast_info -> dist_size_ty option -> dist_size_ty option
    val get_logprob_ts: exp_ty -> dist_size_ty option -> tensor_size_ty
  end

module DistSize : DistSizeTy =
  struct
    exception Cannot_compute_dist_size

    (* converts a given tensor size into a distribution type *)
    let _dist_size_ty_from_tensor_size_ty (dist_kind: dist_kind):
          tensor_size_ty -> dist_size_ty option  = function
      | None -> None
      | Some ts -> 
         begin
           match dist_kind with
           | Normal | Exponential | Gamma | Beta | Uniform _
           | Poisson | Bernoulli | Delta -> 
              Some (ts, [])
           | Dirichlet n_case_opt | OneHotCategorical n_case_opt ->
              Some (list_drop_last 1 ts, [n_case_opt])
           | Categorical _ ->
              Some (list_drop_last 1 ts, [])
           | Subsample (_, subsize_opt) ->
              Some (ts, [subsize_opt]) (* wy: is `ts' always []? *)
         end

    let get_ty (arg_ty_list: exp_ty list) (dist_kind: dist_kind)
        : dist_size_ty option =
      if not (List.for_all is_broadcastable_exp_ty arg_ty_list) then
        None
      else begin
        try
          let tens_size_ty = exp_ty_do_broadcast arg_ty_list in
          _dist_size_ty_from_tensor_size_ty dist_kind tens_size_ty
        with Broadcast_failure ->
          raise (Must_error "compute_dist_size: broadcast failure")
      end

    let _apply_trans (b_size, e_size) = function
      | ToEvent(None) -> raise Cannot_compute_dist_size
      | ToEvent(Some(n)) ->
          let b_len = List.length b_size in
          if b_len < n then
            raise (Must_error
                    ("apply_trans_l: to_event("
                     ^ (string_of_int n)
                     ^ ") called when the batch dim is "
                     ^ (string_of_int b_len)))
          else
            let new_b_size = list_take (b_len - n) b_size in
            let new_e_size = List.rev_append (list_take n (List.rev b_size)) e_size in
            (new_b_size, new_e_size)
      | ExpandBy(delta_size) -> (delta_size @ b_size, e_size)

    let _apply_trans_l (dist_trans_l: dist_trans list)
                       (dist_size: dist_size_ty)
                       : dist_size_ty option =
      try Some (List.fold_left _apply_trans dist_size dist_trans_l)
      with Cannot_compute_dist_size -> None

    let apply_trans_l (dist_trans_l: dist_trans list)
                           : dist_size_ty option -> dist_size_ty option =
      lift_opt (_apply_trans_l dist_trans_l)

    let _apply_broadcast (broadcast_info: broadcast_info)
                         (dist_size: dist_size_ty)
                         : dist_size_ty option =
      try
        let (batch_size, event_size) = dist_size in
        let tens_size_ty0 = Some batch_size in
        let tens_size_ty1 = tensor_size_ty_from_broadcast_info broadcast_info in
        match tensor_size_ty_do_broadcast [tens_size_ty0; tens_size_ty1] with
        | None -> None
        | Some new_dist_size -> Some (new_dist_size, event_size)
      with
      | Cannot_compute_dist_size -> None
      | Broadcast_failure -> raise (Must_error "apply_braodcast: cannot apply broadcast")

    let apply_broadcast (broadcast_info: broadcast_info)
                        : dist_size_ty option -> dist_size_ty option =
      lift_opt (_apply_broadcast broadcast_info)

    let get_logprob_ts (logprob_arg_ty: exp_ty)
          (dist_size_opt: dist_size_ty option): tensor_size_ty =
      match logprob_arg_ty, dist_size_opt with
      | ET_tensor(Some arg_ts_pre), Some (dist_batch_shape, dist_event_shape) ->
         let event_shape_len = List.length dist_event_shape in
         let arg_batch_shape = list_drop_last event_shape_len arg_ts_pre in
         let arg_event_shape = list_take_last event_shape_len arg_ts_pre in
         if dist_event_shape != arg_event_shape then
           raise (Must_error "get_logprob_ts: two event_shapes do not match.")
         else
           (* wy: in fact, we further need to check if
            *     `dist_batch_shape' is broadcastable into `arg_batch_shape'.
            *     If it does not hold, need to raise Must_error. 
            *     For now, we don't raise Must_error, which is sound anyway.*)
           tensor_size_ty_do_broadcast [Some dist_batch_shape;
                                        Some arg_batch_shape]
      | _ -> None
         
  end
