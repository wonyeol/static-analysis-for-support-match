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
 ** adom_apron.ml: Numerical abstract domain based on Apron *)
open Lib
open Ir_sig
open Irty_sig
open Adom_sig
open Apron

module IU = Ir_util


(** Available managers *)
module PA_box =
  (struct
    let module_name = "nd_PA_box"
    type t = Box.t
    let man: t Manager.t =
      Box.manager_alloc ()
  end: APRON_MGR)
module PA_oct =
  (struct
    let module_name = "nd_PA_oct"
    type t = Oct.t
    let man: t Manager.t =
      Oct.manager_alloc ()
  end: APRON_MGR)
module PA_pol =
  (struct
    let module_name = "nd_PA_polka"
    type t = Polka.strict Polka.t
    let man: t Manager.t =
      Polka.manager_alloc_strict ()
  end: APRON_MGR)


(** Helper functions *)

(*
(* Walk through an IR expr and collect all variables *)
let rec expr_collect_vars acc = function
  | Nil | True | False | Num _ | Str _ | StrFmt _ -> acc
  | Name id -> StringSet.add id acc
  | UOp (_, e) -> expr_collect_vars acc e
  | BOp (_, e0, e1) | Comp (_, e0, e1) ->
      expr_collect_vars (expr_collect_vars acc e0) e1
*)

(* Creation of Apron variables *)
let make_apron_var (id: string): Var.t =
  Var.of_string (Printf.sprintf "%s" id)

(* Conversion of an IR expr into an Apron expression
 * (this function is very conservative and rejects many expressions) *)
let make_apron_expr (env: Apron.Environment.t): expr -> Texpr1.t =
  let rec aux (e: expr): Texpr1.t =
    match e with
    | True | False | Str _ | StrFmt _ -> failwith "unhandled expr"
    | Num (Int i) -> Texpr1.cst env (Coeff.s_of_int i)
    | Num (Float f) -> Texpr1.cst env (Coeff.s_of_float f)
    | Name id -> Texpr1.var env (make_apron_var id)
    | BOp (b, e0, e1) ->
        let b =
          match b with
          | Add -> Texpr0.Add
          | Sub -> Texpr0.Sub
          | Mult -> Texpr0.Mul
          | Div -> Texpr0.Div
          | _ -> failwith "binary operator" in
        Texpr1.binop b (aux e0) (aux e1) Texpr1.Real Texpr1.Near
    | _ -> failwith "unhandled expr" in
  aux
let make_apron_cond (env: Apron.Environment.t) (e: expr)
    : Tcons1.t =
  match IU.simplify_exp (fun _ -> ET_unknown) e with
  | True -> (* tautology constraint 0 = 0 *)
      let ex = make_apron_expr env (Num (Int 0)) in
      Tcons1.make ex Tcons1.EQ
  | False -> (* anti-tautology constraint 1 = 0 *)
      let ex = make_apron_expr env (Num (Int 1)) in
      Tcons1.make ex Tcons1.EQ
  | Comp (c, e0, e1) ->
      (*   e0 (c) e1    is translated into    e0 - e1 (c) 0   *)
      let c, e0, e1 =
        match c with
        | Eq    -> Lincons1.EQ   , e0, e1
        | NotEq -> Lincons1.DISEQ, e0, e1
        | Is    -> Lincons1.EQ   , e0, e1
        | NotIs -> Lincons1.DISEQ, e0, e1
        | Lt    -> Lincons1.SUP  , e1, e0
        | LtE   -> Lincons1.SUPEQ, e1, e0
        | Gt    -> Lincons1.SUP  , e0, e1
        | GtE   -> Lincons1.SUPEQ, e0, e1 in
      let ex = make_apron_expr env (BOp (Sub, e0, e1)) in
      Tcons1.make ex c
  | _ ->
      (* todo *)
      let ex = make_apron_expr env (Num (Int 0)) in
      Tcons1.make ex Tcons1.EQ
        (* failwith (Printf.sprintf "make_apron_cond: %s"
           (IU.expr_to_string e)) *)

(* Make thresholds to use for widening *)
(* xr: this is a bit experimental now so I consider only one
 * expression for now, but we can generalize later *)
let make_thr env (el: expr list): Lincons1.earray =
  match el with
  | [ e ] ->
      begin
        match e with
        | Comp (Lt, Name x, Name y) ->
            let ea = Lincons1.array_make env 2 in
            let le = Linexpr1.make env in
            Linexpr1.set_list le
              [ Coeff.s_of_int (-1), make_apron_var x ;
                Coeff.s_of_int   1 , make_apron_var y ]
              (Some (Coeff.s_of_int 0));
            (* threshold -x + y >  0 *)
            let lcgt = Lincons1.make le Lincons1.SUP
            (* threshold -x + y >= 0 *)
            and lcge = Lincons1.make le Lincons1.SUPEQ in
            Lincons1.array_set ea 0 lcgt;
            Lincons1.array_set ea 1 lcge;
            ea
        | _ ->
            failwith (Printf.sprintf "todo:make_thr:%s"
                        (IU.expr_to_string e))
      end
  | _ -> failwith "several expresions"

(* Apron constraint extraction *)
let extract_coeff_from_cons v cons =
  try Lincons1.get_coeff cons v
  with exn -> failwith "extract_coeff_from_cons"

(* Apron pretty-printing *)
let coeff_2str (c: Coeff.t): string =
  match c with
  | Coeff.Scalar scal -> Scalar.to_string scal
  | Coeff.Interval _ -> failwith "pp_coeff-interval"
let cons_trailer_2str (typ: Lincons0.typ): string =
  match typ with
  | Lincons1.EQ    -> " = 0"
  | Lincons1.DISEQ -> " != 0"
  | Lincons1.SUP   -> " > 0"
  | Lincons1.SUPEQ -> " >= 0"
  | Lincons1.EQMOD s -> Printf.sprintf " == 0 (%s)" (Scalar.to_string s)

(* Pretty-printing of an array of Apron constraints *)
let buf_linconsarray (buf: Buffer.t) (a: Lincons1.earray): unit =
  (* extraction of the integer variables *)
  let env = a.Lincons1.array_env in
  let ivars, fvars = Environment.vars env in
  (* pretty-printing of a constraint *)
  let f_lincons (cons: Lincons1.t): unit =
    if Lincons1.is_unsat cons then Printf.bprintf buf "UNSAT\n"
    else
      (* print non zero coefficients *)
      let mt = ref false in
      Array.iter
        (fun v ->
          let c = extract_coeff_from_cons v cons in
          if not (Coeff.is_zero c) then
            let vname = Var.to_string v in
            if !mt then Printf.bprintf buf " + "
            else mt := true;
            Printf.bprintf buf "%s . %s" (coeff_2str c) vname
        ) fvars;
      (* print the constant *)
      let d0 = coeff_2str (Lincons1.get_cst cons) in
      Printf.bprintf buf "%s%s" (if !mt then " + " else "") d0;
      (* print the relation *)
      Printf.bprintf buf "%s\n" (cons_trailer_2str (Lincons1.get_typ cons)) in
  (* Array of cons1 *)
  let ac1 =
    Array.mapi
      (fun i _ -> Lincons1.array_get a i)
      a.Lincons1.lincons0_array in
  Array.iter f_lincons ac1

(* Scalar.t --> float --> int *)
let rec scalar_to_float (round: Mpfr.round) (s: Scalar.t) : float =
  (* Mpfr.round = Near | Zero | Up | Down *)
  match s with
  | Scalar.Float f -> f
  | Scalar.Mpfrf m -> Mpfrf.to_float ~round:round m
  | Scalar.Mpqf m -> scalar_to_float round (Scalar.Mpfrf (Mpfrf.of_mpq m round))

let float_to_int (round: Mpfr.round) (f: float): int =
  match round with
  | Mpfr.Up   -> int_of_float (ceil  f)
  | Mpfr.Down -> int_of_float (floor f)
  | _ -> failwith "float_to_int: unimplemented case"

let scalar_to_int (round: Mpfr.round) (s: Scalar.t): int =
  float_to_int round (scalar_to_float round s)

(** Apron domain constructor *)
module Make = functor (M: APRON_MGR) ->
  (struct
    let module_name = "Apron(" ^ M.module_name ^ ")"

    module A = Apron.Abstract1
    let man = M.man

    (* Abstract values:
     * - an enviroment
     * - and a conjunction of constraints in Apron representation (u) *)
    type t = M.t A.t

    let is_bot t = A.is_bottom man t

    (* Pp *)
    let buf_t (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "%a" buf_linconsarray (A.to_lincons_array man t)
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    (* tries to prove that a condition holds
     * soundness: if sat e t returns true, all states in gamma(t) satisfy e *)
    let sat (e: expr) (t: t): bool =
      let env = A.env t in
      let ce = make_apron_cond env e in
      A.sat_tcons man t ce

    (* Evaluation of commands *)
    let rec eval ac u =
      match ac with
      | Assert e ->
          if sat e u then
            u
          else
            failwith (Printf.sprintf "%s.eval: Cannot prove Assert" module_name)
      | Assume e ->
          (* convert the expression to Apron IR *)
          let env = A.env u in
          let ce = make_apron_cond env e in
          let eacons = Tcons1.array_make env 1 in
          Tcons1.array_set eacons 0 ce;
          (* perform the condition test *)
          let u = A.meet_tcons_array man u eacons in
          (* red bottom *)
          if is_bot u then raise Bottom
          else u

      | Assn (x,e) ->
          (* convert the expression to Apron IR *)
          let lv = make_apron_var x
          and rv = make_apron_expr (A.env u) e in
          (* perform the Apron assignment *)
          A.assign_texpr_array man u [| lv |] [| rv |] None

      | AssnCall _ ->
          failwith "todo:eval:assncall"

      | Sample _ ->
          failwith "todo:eval:sample"

    let enter_with withitem u =
      failwith "todo:enter_with:adom_apron"

    let exit_with withitem u =
      failwith "todo:exit_with:adom_apron"

    (* Lattice elements and operations *)
    let top =
      let env_empty = Environment.make [| |] [| |] in
      A.top man env_empty
    let init_t = top

    let join = A.join man
    let widen thr x0 x1 =
      if thr = [ ] then A.widening man x0 x1
      else A.widening_threshold man x0 x1 (make_thr (A.env x0) thr)
    let leq (u0: t) (u1: t): bool =
      A.is_leq man u0 u1

    (* Dimensions management *)
    (* these functions are not too hard to implement, but first we need
     * to make sure the interfaces are ok *)
    let dim_add (dn: string) (t: t): t =
      let var = make_apron_var dn in
      let env_old = A.env t in
      let env_new =
        try Environment.add env_old [| |] [| var |]
        with e ->
          failwith (Printf.sprintf "dim_add: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_rem_set (dns: SS.t) (t: t): t =
      let lvars = SS.fold (fun dn l -> make_apron_var dn :: l) dns [ ] in
      let env_old = A.env t in
      let env_new =
        try Environment.remove env_old (Array.of_list lvars)
        with e ->
          failwith (Printf.sprintf "dim_rem: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_rem (dn: string) (t: t): t =
      let var = make_apron_var dn in
      let env_old = A.env t in
      let env_new =
        try Environment.remove env_old [| var |]
        with e ->
          failwith (Printf.sprintf "dim_rem: %s" (Printexc.to_string e)) in
      A.change_environment man t env_new false
    let dim_mem (dn: string) (t: t): bool =
      let var = make_apron_var dn in
      let env = A.env t in
      try Environment.mem_var env var
      with e ->
        failwith (Printf.sprintf "dim_mem: %s" (Printexc.to_string e))
    let dim_project_out (dn: string) (t: t): t =
      let var = make_apron_var dn in
      A.forget_array man t [| var |] false
    let dims_get (t: t): SS.t option =
      let env = A.env t in
      let ai, af = Environment.vars env in
      let r = ref SS.empty in
      let f v = r := SS.add (Var.to_string v) !r in
      Array.iter f ai;
      Array.iter f af;
      Some !r

    (* ad-hoc function *)
    let set_aux_distty vto t =
      failwith "adom_apron.ml: set_aux_distty must not be called!"

    let bound_var_apron (dn: string) (t: t): int*int =
        let open Interval in
        let intvl : Interval.t = A.bound_variable man t (make_apron_var dn) in
        let inf : int = scalar_to_int Mpfr.Down intvl.inf in
        let sup : int = scalar_to_int Mpfr.Up   intvl.sup in
        (inf, sup)

    (* checks whether a domain-specific relationship holds
     * between two abstract states *)
    let is_related t1 t2 = false

    let range_info e t =
      failwith "todo:range_info:adom_apron"
  end: ABST_DOMAIN_NB_D)
