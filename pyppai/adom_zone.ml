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
 ** adom_zone.ml: implementation of functors adding zone information
 **)
open Lib
open Ir_sig
open Adom_sig

(** bnd_expr *)
let expr_to_bnd_expr: expr -> bnd_expr = function
  | Num(Int(n)) -> BE_cst n
  | Name v -> BE_var v
  | _ -> failwith "expr_to_bnd_expr: cannot translate"

(** "Identity" functor above numerical domain;
 ** Adds no zone predicate at all (same as using the old numerical domain) *)
module MakeId (D: ABST_DOMAIN_NB_D) =
  (struct
    include D
    let zone_dim_add _ (x: t): t = x (*failwith "ID: dim_add"*)
    let zone_dim_rem _ (x: t): t = x (*failwith "ID: dim_rem"*)
    let zone_add_cell _ _ (x: t): t = failwith "ID: add_cell"
    let zone_sat _ _ (x: t): bool = false
    let zone_include _ _ (x: t): bool = false
    let zone_is_related _ _ _: bool = false
  end: ABST_DOMAIN_ZONE_NB)

(** Functor adding zone predicates *)
module MakeZone (D: ABST_DOMAIN_NB_D) =
  (struct
    let module_name = D.module_name ^ "+zone"
    let debug = false

    (* Bound:
     * - a bount may comprise at most two constraints;
     * - one of the form b=cst
     * - and one either of the form b=var or b=var-1;
     * - if it has neither of these two forms of constraints, it can be
     *   pruned away *)
    type bound =
        { b_cst:    int option;
          b_var:    (string * bool) option; (* true => -1; false => 0 *) }

    (* Description of a zone in a tensor:
     *  - tzone: a tensor zone is a disjunction of single zones
     *           plus a boolean true if and only if it is an exact abstraction
     *  - ozone: a single zone is a list of ranges (one per dimension)
     *  - range: a range is a pair of bounds
     *  - bound: a bound is a list of bound values that are equal *)
    type range = bound * bound
    type ozone = range list
    type tzone =
        { tz_oz: ozone list;  (* the disjunctive zone description *)
          tz_ex: bool;        (* whether it is an exact description *) }

    (* Abstract elements *)
    type t = { t_u:  D.t ;
               t_z:  tzone StringMap.t }

    (* Printing *)
    let buf_bnd_expr (buf: Buffer.t): bnd_expr -> unit = function
      | BE_cst i -> Printf.bprintf buf "%d" i
      | BE_var v -> Printf.bprintf buf "%s" v
      | BE_var_minus_one v -> Printf.bprintf buf "%s-1" v
    let buf_bound (buf: Buffer.t) (b: bound): unit =
      let pp_var (buf: Buffer.t) (v, b) =
        Printf.bprintf buf "%s%s" v (if b then "-1" else "") in
      match b.b_cst, b.b_var with
      | None, None -> Printf.bprintf buf "___"
      | None, Some v -> pp_var buf v
      | Some i, None -> Printf.bprintf buf "%d" i
      | Some i, Some v -> Printf.bprintf buf "%d=%a" i pp_var v
    let buf_range (buf: Buffer.t) (bmin, bmax): unit =
      Printf.bprintf buf "[%a,%a]" buf_bound bmin buf_bound bmax
    let buf_ozone: Buffer.t -> ozone -> unit = buf_list " X " buf_range
    let buf_tzone (buf: Buffer.t) (tz: tzone): unit =
      buf_list " U " buf_ozone buf tz.tz_oz;
      Printf.bprintf buf "   [%s]" (if tz.tz_ex then "exact" else "approx")
    let buf_t buf t =
      D.buf_t buf t.t_u;
      StringMap.iter
        (fun v tz ->
          Printf.bprintf buf "%s => %a\n" v buf_tzone tz
        ) t.t_z
    let pp_bound = buf_to_channel buf_bound
    let pp_range = buf_to_channel buf_range
    let pp_ozone = buf_to_channel buf_ozone
    let pp_tzone = buf_to_channel buf_tzone
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    (* Helper functions on zones *)
    (* Top element in tzone type *)
    let tz_top = { tz_oz = [ ]; tz_ex = false }
    (* Ignored functions (warns + information soundly dropped) *)
    let tz_u_drop msg t =
      if t.t_z = StringMap.empty then t
      else
        let m = StringMap.map (fun _ -> { tz_oz = [ ] ;
                                          tz_ex = false }) t.t_z in
        Printf.printf "WARN, Zone information dropped (%s)\n" msg;
        { t with t_z = m }
    (* Filter zone information, when a dimension is dropped or assigned
     * a value that we cannot precisely bound *)
    exception Drop_zone of string
    let drop_empty_bound b =
      if b.b_cst = None && b.b_var = None then raise (Drop_zone "emp bound")
      else b
    let mod_bound_in_tzone (f: bound -> bound) (tz: tzone): tzone =
      let aux_bound b = drop_empty_bound (f b) in
      let aux_range (bmin, bmax) = aux_bound bmin, aux_bound bmax in
      let aux_ozone = List.map aux_range in
      { tz with tz_oz = List.map aux_ozone tz.tz_oz }
    let tzone_filter_modvar (d: string): tzone -> tzone =
      let aux_bound b =
        match b.b_var with
        | Some (v, o) -> if v = d then { b with b_var = None } else b
        | None -> b in
      mod_bound_in_tzone aux_bound
    let incr_var_one (d: string): tzone -> tzone =
      let aux_bound b =
        match b.b_var with
        | Some (v, o) ->
            if v = d then
              if o then drop_empty_bound { b with b_var = None }
              else { b with b_var = Some (v, true) }
            else b
        | None -> b in
      mod_bound_in_tzone aux_bound
    let tzone_var_constrain (d: string) (n: int): tzone -> tzone =
      let aux_bound b =
        match b.b_var with
        | Some (v, o) ->
            if v = d then
              match b.b_cst with
              | None -> { b with b_cst = Some n }
              | Some i ->
                  if o && i = n-1 || not o && i = n then b
                  else raise Bottom
            else b
        | None -> b in
      mod_bound_in_tzone aux_bound
    let tzone_update_all (f: tzone -> tzone) (m: tzone StringMap.t)
        : tzone StringMap.t =
      StringMap.map
        (fun tz ->
          try f tz
          with Drop_zone _ -> tz_top
        ) m
    (* Making new zone predicates (initialization, zone assertion...) *)
    let ozone_make (t: t) (l: (bnd_expr * bnd_expr) list): ozone =
      let mk_bound (be: bnd_expr): bound =
        match be with
        | BE_cst i -> { b_cst = Some i ; b_var = None }
        | BE_var v ->
            let bmin, bmax = D.bound_var_apron v t.t_u in
            let ocst = if bmin = bmax then Some bmin else None in
            { b_cst = ocst; b_var = Some (v, false) }
        | BE_var_minus_one v ->
            let bmin, bmax = D.bound_var_apron v t.t_u in
            let ocst = if bmin = bmax then Some (bmin-1) else None in
            { b_cst = ocst; b_var = Some (v, true) } in
      let mk_range (b: bnd_expr * bnd_expr): range =
        mk_bound (fst b), mk_bound (snd b) in
      List.map mk_range l
    (* Comparison of zones
     *  (returns false when any of the arguments is a may) *)
    let tzone_eq (tz0: tzone) (tz1: tzone): bool =
      let bound_full_eq b0 b1 =
        match b0.b_cst, b1.b_cst with
        | Some i0, Some i1 -> i0 = i1
        | _, _ ->
            match b0.b_var, b1.b_var with
            | Some (v0,m0), Some (v1,m1) -> v0=v1 && m0=m1
            | _, _ -> false in
      let bound_eq b0 b1 =
        b0 = b1 || bound_full_eq b0 b1 in
      let ozone_eq: ozone -> ozone -> bool =
        let range_eq (bmin0, bmax0) (bmin1, bmax1) =
          bound_eq bmin0 bmin1 && bound_eq bmax0 bmax1 in
        List.fold_left2 (fun acc r0 r1 -> acc && range_eq r0 r1) true in
      try
        if false then
          Printf.printf "compare:\n%a\n%a\nex: %b %b\n"
            pp_tzone tz0 pp_tzone tz1 tz0.tz_ex tz1.tz_ex;
        List.fold_left2 (fun acc oz0 oz1 -> acc && ozone_eq oz0 oz1)
          (tz0.tz_ex && tz1.tz_ex) tz0.tz_oz tz1.tz_oz
      with Invalid_argument _ -> false

    (* Variable modification *)
    let tzone_var_mod (x: string) (tz: tzone): tzone =
      let module M = struct exception Drop end in
      let do_bound b =
        match b.b_var with
        | None -> b
        | Some (v, _) ->
            if v = x then
              if b.b_cst = None then raise M.Drop
              else b
            else b in
      let do_range (b0, b1) = do_bound b0, do_bound b1 in
      let do_ozone (oz: ozone): ozone = List.map do_range oz in
      try
        let ozl = List.map do_ozone tz.tz_oz in
        { tz with tz_oz = ozl }
      with M.Drop ->
        { tz_oz = [ ];
          tz_ex = true }

    (* Condensing zones:
     *  -> trying to make contiguous ozones a single ozone inside a tzone
     *     (this is a form of reduction, which makes representation more
     *     compact and easier to manipulate) *)
    exception NotMergeable
    let rec tzone_condense (tz: tzone): tzone =
      let bound_eq b0 b1 =
        let rc, nc =
          match b0.b_cst, b1.b_cst with
          | Some i0, Some i1 ->
              let c = i0 = i1 in
              c, if c then b0.b_cst else None
          | _, _ -> false, None in
        let rv, nv =
          match b0.b_var, b1.b_var with
          | Some (v0, s0), Some (v1, s1) ->
              let c = v0 = v1 && s0 = s1 in
              c, if c then b0.b_var else None
          | _, _ -> false, None in
        if debug then
          Printf.printf "bound_eq %a,%a => %b, %b (%b)\n"
            pp_bound b0 pp_bound b1 rc rv (b0 = b1);
        rc || rv, { b_cst = nc ; b_var = nv } in
      let bound_is_prev b0 b1 =
        let rc =
          match b0.b_cst, b1.b_cst with
          | Some i0, Some i1 -> i0 = i1 - 1
          | _, _ -> false in
        let rv =
          match b0.b_var, b1.b_var with
          | Some (v0, true), Some (v1, false) -> v0 = v1
          | _, _ -> false in
        if debug then
          Printf.printf "bound_next: %a, %a => %b, %b\n"
            pp_bound b0 pp_bound b1 rc rv;
        rc || rv in
      let ozone_condense =
        let rec aux maymerge oz0 oz1 =
          match oz0, oz1 with
          | (bmin0, bmax0) :: ozo0, (bmin1, bmax1) :: ozo1 ->
              begin
                match bound_eq bmin0 bmin1, bound_eq bmax0 bmax1 with
                | (true, bmin), (true, bmax) ->
                    (bmin, bmax) :: aux maymerge ozo0 ozo1
                | _, _ ->
                    if maymerge then
                      if bound_is_prev bmax0 bmin1 then
                        (bmin0, bmax1) :: aux false ozo0 ozo1
                      else if bound_is_prev bmax1 bmin0 then
                        (bmin1, bmax0) :: aux false ozo0 ozo1
                      else raise NotMergeable
                    else raise NotMergeable
              end
          | [ ], [ ] -> [ ]
          | _, _ -> failwith "tensors of incompatible dimensions" in
        aux true in
      match tz.tz_oz with
      | [ ] | [ _ ] -> tz
      | oz0 :: oz1 :: l ->
          (* very restrictive, ok for now *)
          try tzone_condense { tz with tz_oz = ozone_condense oz0 oz1 :: l }
          with NotMergeable ->
            let ctz = tzone_condense { tz with tz_oz = oz1 :: l } in
            { ctz with tz_oz = oz0 :: ctz.tz_oz }
    let reduce_condense (t: t): t =
      { t with t_z = StringMap.map tzone_condense t.t_z }

    (* Lattice values *)
    let top: t =
      { t_u = D.top;
        t_z = StringMap.empty }
    let is_bot (t: t) = D.is_bot t.t_u
    let init_t: t = top

    (* Lattice operations *)
    (* Helper functions for lattice operations *)
    let tzone_unify (tz0: tzone) (tz1: tzone): tzone =
      let bound_unify b0 b1 =
        let option_map2 f o0 o1 =
          match o0, o1 with
          | Some x0, Some x1 -> f x0 x1
          | _, _ -> None in
        let rc =
          option_map2 (fun x y -> if x = y then Some x else None)
            b0.b_cst b1.b_cst
        and rv =
          option_map2 (fun x y -> if x = y then Some x else None)
            b0.b_var b1.b_var in
        match rc, rv with
        | None, None -> raise (Drop_zone "unify")
        | _, _ -> { b_cst = rc ;
                    b_var = rv } in
      let range_unify (bmin0, bmax0) (bmin1, bmax1) =
        bound_unify bmin0 bmin1, bound_unify bmax0 bmax1 in
      if debug then
        Printf.printf "unification:\n%a\n%a\n" pp_tzone tz0 pp_tzone tz1;
      let ozone_unify = List.map2 range_unify in
      { tz_oz = List.map2 ozone_unify tz0.tz_oz tz1.tz_oz;
        tz_ex = tz0.tz_ex && tz1.tz_ex }
    let mtzone_join (op: string)
        (z0: tzone StringMap.t) (z1: tzone StringMap.t): tzone StringMap.t =
      let pp _ =
        StringMap.iter (fun v z -> Printf.printf "    %s: %a\n" v pp_tzone z) in
      if debug then
        Printf.printf "Before condense:\n -Z0\n%a -Z1\n%a" pp z0 pp z1;
      let z0 = StringMap.map tzone_condense z0 in
      let z1 = StringMap.map tzone_condense z1 in
      if debug then
        Printf.printf "After condense:\n -Z0\n%a -Z1\n%a" pp z0 pp z1;
      let z =
        let z =
          StringMap.fold
            (fun v _ acc ->
              if StringMap.mem v z0 then acc
              else StringMap.add v tz_top acc
            ) z1 StringMap.empty in
        StringMap.fold
          (fun v tz0 acc ->
            try
              let tz1 = StringMap.find v z1 in
              StringMap.add v (tzone_unify tz0 tz1) acc
            with Not_found | Drop_zone _ ->
              StringMap.add v tz_top acc
          ) z0 z in
      if debug then
        begin
          Printf.printf "WARN, %s, check result\n" op;
          Printf.printf "Merging results-l:\n";
          StringMap.iter (fun v -> Printf.printf "%s => %a\n" v pp_tzone) z0;
          Printf.printf "Merging results-r:\n";
          StringMap.iter (fun v -> Printf.printf "%s => %a\n" v pp_tzone) z1;
          Printf.printf "Unify results-r:\n";
          StringMap.iter (fun v -> Printf.printf "%s => %a\n" v pp_tzone) z;
        end;
      z
    let join (t0: t) (t1: t): t =
      { t_u = D.join t0.t_u t1.t_u;
        t_z = mtzone_join "join" t0.t_z t1.t_z }
    let widen el (t0: t) (t1: t): t =
      { t_u  = D.widen el t0.t_u t1.t_u;
        t_z = mtzone_join "widen" t0.t_z t1.t_z }
    let leq (t0: t) (t1: t): bool =
      let module M = struct exception Stop end in
      let zones_leq z0 z1 =
        try
          StringMap.iter
            (fun v tz1 ->
              if not (tzone_eq (StringMap.find v z0) tz1) then raise M.Stop
            ) z1;
          true
        with Not_found | M.Stop -> false in
      D.leq t0.t_u t1.t_u && (zones_leq t0.t_z t1.t_z)

    (* Post-conditions for abstract commands *)
    let eval (c: acmd) (t: t): t =
      let drop (msg: string) =
        tz_u_drop (Printf.sprintf "eval %s: %s" msg (Ir_util.acmd_to_string c))
          { t with t_u = D.eval c t.t_u; } in
      match c with
      | Assn (vl, BOp (Add, Num (Int 1), Name vr))
      | Assn (vl, BOp (Add, Name vr, Num (Int 1))) ->
          let t = reduce_condense t in
          if vl = vr then
            { t_u = D.eval c t.t_u;
              t_z = tzone_update_all (incr_var_one vl) t.t_z }
          else drop "assgn(x=y+1)"
      | Assn (vl, Num (Int _)) ->
          { t_u = D.eval c t.t_u ;
            t_z = tzone_update_all (tzone_filter_modvar vl) t.t_z }
      | Assn (vl, _) ->
          { t_u = D.eval c t.t_u ;
            t_z = StringMap.map (tzone_var_mod vl) t.t_z }
      | Assume (Comp (LtE, Name v, BOp (Sub, Num (Int _), Num (Int _))))
      | Assume (Comp (GtE, Name v, Num _))
      | Assume (Comp (LtE, Name v, Num _)) ->
          (* xr: we could simplify the conditions beforehand... *)
          let t = { t with t_u = D.eval c t.t_u } in
          let bmin, bmax = D.bound_var_apron v t.t_u in
          if bmin = bmax then
            (* add a constraint *)
            { t with
              t_z = tzone_update_all (tzone_var_constrain v bmin) t.t_z }
          else
            (* no constraint to add;
             * for now, no reduction, but we could consider *)
            t
      | Assert _ -> { t with t_u = D.eval c t.t_u }
      | _ -> drop "other"

    (* With clause *)
    let enter_with (wi: withitem) (t: t): t =
      tz_u_drop "enter_with" { t with t_u = D.enter_with wi t.t_u }
    let exit_with (wi: withitem) (t: t): t =
      tz_u_drop "exit_with" { t with t_u = D.exit_with wi t.t_u }

    (* Condition checking *)
    let sat (e: expr) (t: t): bool = D.sat e t.t_u

    (* Whether two dimensions are related *)
    let is_related (t0: t) (t1: t): bool = D.is_related t0.t_u t1.t_u

    (* Dimensions management *)
    let dim_add s t =
      { t with t_u = D.dim_add s t.t_u }
    let dim_rem s t =
      { t_u = D.dim_rem s t.t_u;
        t_z = tzone_update_all (tzone_filter_modvar s) t.t_z }
    let dim_project_out s t =
      { t_u = D.dim_project_out s t.t_u;
        t_z = tzone_update_all (tzone_filter_modvar s) t.t_z }
    let dim_mem s t = D.dim_mem s t.t_u
    let dims_get t = D.dims_get t.t_u

    (* Ad-hoc functions used by only some modules *)
    let set_aux_distty o (t: t): t =
      failwith "adom_zone.ml: set_aux_distty must not be called!"
    let bound_var_apron (s: string) (t: t): int * int =
      D.bound_var_apron s t.t_u

    (* Zone operations *)
    let zone_dim_add (s: string) (t: t): t =
      let tz = { tz_oz = [ ];
                 tz_ex = true } in
      { t with t_z = SM.add s tz t.t_z }
    let zone_dim_rem (s: string) (t: t): t =
      { t with t_z = SM.remove s t.t_z }
    (* xr: need to saturate the constraints... *)
    let zone_add_cell (s: string) (z: bnd_expr list) (t: t): t =
      let nz = ozone_make t (List.map (fun b -> b, b) z) in
      let zone =
        try
          let tz = StringMap.find s t.t_z in
          tzone_condense { tz with tz_oz = nz :: tz.tz_oz }
        with Not_found -> { tz_oz = [ nz ];
                            tz_ex = true } in
      { t with t_z = StringMap.add s zone t.t_z }
    let zone_sat (s: string) (z: (bnd_expr * bnd_expr) list) (t: t) =
      try
        let refz = ozone_make t z in
        let curz = StringMap.find s t.t_z in
        tzone_eq curz { tz_oz = [ refz ];
                        tz_ex = true }
      with Not_found -> false

    (* zone_include *)
    (* e <= b always? true means yes, false means i don't know. *)
    let bound_leq (t: t) (e: bnd_expr) (b: bound): bool =
      match e, b.b_var with
      | BE_var v, Some(w, false) when v=w -> true
      | _ ->
         let e_max =
           match e with
           | BE_cst n -> n
           | BE_var v           -> snd (D.bound_var_apron v t.t_u)
           | BE_var_minus_one v -> snd (D.bound_var_apron v t.t_u) - 1 in
         let b_min_opt =
           match b.b_cst, b.b_var with
           | Some n, _ -> Some n
           | _, Some (v, false) -> Some (fst (D.bound_var_apron v t.t_u))
           | _, Some (v, true)  -> Some (fst (D.bound_var_apron v t.t_u) - 1)
           | None, None -> None in
         match b_min_opt with
         | Some b_min -> e_max <= b_min
         | _ -> false

    (* e >= b always? true means yes, false means i don't know. *)
    let bound_geq (t: t) (e: bnd_expr) (b: bound): bool =
      match e, b.b_var with
      | BE_var v, Some(w, _) when v=w -> true
      | _ ->
         let e_min =
           match e with
           | BE_cst n -> n
           | BE_var v           -> fst (D.bound_var_apron v t.t_u)
           | BE_var_minus_one v -> fst (D.bound_var_apron v t.t_u) - 1 in
         let b_max_opt =
           match b.b_cst, b.b_var with
           | Some n, _ -> Some n
           | _, Some (v, false) -> Some (snd (D.bound_var_apron v t.t_u))
           | _, Some (v, true)  -> Some (snd (D.bound_var_apron v t.t_u) - 1)
           | None, None -> None in
         match b_max_opt with
         | Some b_max -> e_min >= b_max
         | _ -> false

    let zone_include (z: bnd_expr list) (s: string) (t: t): bool =
      let range_include (e: bnd_expr) ((lb,ub): range): bool =
        (bound_geq t e lb) && (bound_leq t e ub) in
      let ozone_include: bnd_expr list -> ozone -> bool =
        List.fold_left2 (fun acc e0 r0 -> acc && range_include e0 r0) true in
      let tzone_include (es: bnd_expr list) (tz: tzone): bool =
        List.fold_left (fun acc o0 -> acc || ozone_include es o0)
          false tz.tz_oz in
      try tzone_include z (SM.find s t.t_z)
      with Not_found -> false

    (* is_related_zone *)
    let zone_is_related (s: string) (t1: t) (t2: t): bool =
      (* wy: for debug.
      let _ =
        Printf.printf "zone_is_related: s = %s\n" s;
        Printf.printf "  t1 = \n%a\n" pp t1;
        Printf.printf "  t2 = \n%a\n" pp t2 in *)
      let tzone1, tzone2 =
        try SM.find s t1.t_z, SM.find s t2.t_z
        with Not_found -> failwith "zone_is_related: have different dim" in
      tzone_eq tzone1 tzone2

    let range_info e t =
      failwith "todo:range_info:adom_zone"
  end: ABST_DOMAIN_ZONE_NB)
