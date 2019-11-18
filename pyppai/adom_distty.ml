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
 ** adom_distty.ml: implementation of a distribution type domain
 **)
open Lib
open Ir_sig
open Irty_sig
open Adom_sig

module IU = Ir_util
module ITU = Irty_util

(** Module *)
module DistType =
  (struct
    let module_name = "DistType"

    (* invariant:
     *  dom(t_dist) = t_samp
     *)
    type dist_lat = DBot | DSome of dist_kind | DTop
    type vtyp_lat = VBot | VSome of vtyp | VTop
    type aux_ty = vtyp option
    type t = {
      t_samp: SS.t;
      t_dist: (dist_lat * vtyp_lat) SM.t;
      t_aux : aux_ty;
      (* t_aux has no meaning at all, except in the eval function.
       * In eval, t_aux of the argument of type t denotes vtyp of
       * the resulting value from Sample expression, where None denotes Top
       * in (vtyp+Top).
       * We add t_aux to t to pass vtyp value to the body of eval function.
       * We think that this would be the simplest solution;
       * the other solution is to use a different signature only for this
       * module, but this would make our codes ugly. *)
    }

    (* lattice values *)
    let top = {
      t_samp = SS.empty;
      t_dist = SM.empty;
      t_aux  = None (* this value has no meaning at all *);
    }
    let is_bot (t: t) = false
    let init_t: t = top

    (* printing *)
    let buf_dist_lat (buf: Buffer.t) : dist_lat -> unit = function
      | DBot    -> Printf.bprintf buf "_|_"
      | DSome d -> Printf.bprintf buf "%s" (IU.dist_kind_to_string d)
      | DTop    -> Printf.bprintf buf "T"
    let buf_vtyp_lat (buf: Buffer.t) : vtyp_lat -> unit = function
      | VBot     -> Printf.bprintf buf "_|_"
      | VSome vt -> Printf.bprintf buf "%s" (ITU.vtyp_to_string vt)
      | VTop     -> Printf.bprintf buf "T"
    let buf_t (buf: Buffer.t) (t: t) : unit =
      if is_bot t then Printf.bprintf buf "_|_"
      else begin
          Printf.bprintf buf "[";
          List.iter (fun (s,(d,v)) ->
                     Printf.bprintf buf "%s->(%a,%a), " s
                                    buf_dist_lat d
                                    buf_vtyp_lat v)
                    (SM.bindings t.t_dist);
          Printf.bprintf buf "]"
        end
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    (* lattice operations *)
    let join_dist_lat (d1: dist_lat) (d2: dist_lat) =
      match d1, d2 with
      | DBot, d | d, DBot -> d
      | DSome dist1, DSome dist2 when dist1 = dist2 -> d1
      | _ -> DTop
    let leq_dist_lat (d1: dist_lat) (d2: dist_lat) =
      match d1, d2 with
      | DBot, _ | _, DTop -> true
      | DSome dist1, DSome dist2 -> dist1 = dist2
      | _ -> false

    let join_vtyp_lat (v1: vtyp_lat) (v2: vtyp_lat) =
      match v1, v2 with
      | VBot, v | v, VBot -> v
      | VSome (Vt_num nt1), VSome (Vt_num nt2) ->
         VSome (Vt_num (ITU.num_ty_join nt1 nt2))
      | VSome (Vt_tens ts1), VSome (Vt_tens ts2) ->
         VSome (Vt_tens (ITU.tensor_size_ty_join ts1 ts2))
      | _ -> VTop
    let leq_vtyp_lat (v1: vtyp_lat) (v2: vtyp_lat) =
      match v1, v2 with
      | VBot, _ | _, VTop -> true
      | VSome vt1, VSome vt2 -> ITU.vtyp_leq vt1 vt2
      | _ -> false

    let join (t1: t) (t2: t) : t =
      let join_each s dopt1 dopt2 =
        match dopt1, dopt2 with
        | Some (d1,v1), Some (d2,v2) -> Some (join_dist_lat d1 d2,
                                              join_vtyp_lat v1 v2)
        | _ -> failwith "join: t1.t_samp != t2.t_samp" in
      let dist_new = SM.merge join_each t1.t_dist t2.t_dist in
      { t_samp = t1.t_samp;
        t_dist = dist_new;
        t_aux  = top.t_aux; }
    let widen (thr: expr list): t -> t -> t = join
    let leq (t1: t) (t2: t) : bool =
      if not (SS.equal t1.t_samp t2.t_samp) then
        failwith "leq: t1.t_samp != t2.t_samp"
      else
         let leq_each (s: SS.elt) (b: bool) : bool =
           let (d1,v1) = SM.find s t1.t_dist in
           let (d2,v2) = SM.find s t2.t_dist in
           b && (leq_dist_lat d1 d2) && (leq_vtyp_lat v1 v2)
         in SS.fold leq_each t1.t_samp true

    (* helper function *)
    let aux_to_vtyp_lat (a: aux_ty): vtyp_lat =
      match a with
      | None -> VTop
      | Some vt -> VSome vt
    (* post-conditions for abstract commands *)
    let rec eval (ac: acmd) (t: t) =
      (* Require: t.t_aux must be set before this function is called.
       * Here t.t_aux denotes vtyp of the value returned by Sample expr. *)
      match ac with
      | Sample (_, name, (dist_kind, _(*, _*)), _, obs_opt) ->
         begin
           match name with
           | Str s | StrFmt (s, _) ->
              let (d_prev, v_prev) =
                try SM.find s t.t_dist
                with Not_found ->
                  failwith (Printf.sprintf "Format string %s not found" s) in
              let d_cur = 
                match obs_opt with 
                | None | Some Nil -> (DSome dist_kind) 
                | Some _ -> DTop in
              let d_new = join_dist_lat d_prev d_cur in
              let v_new = join_vtyp_lat v_prev (aux_to_vtyp_lat t.t_aux) in
              { t with t_dist = SM.add s (d_new, v_new) t.t_dist }
           | _ -> t
         end
      | _ -> t

    let enter_with withitem t =
      failwith "todo:enter_with:adom_distty"

    let exit_with withitem t =
      failwith "todo:exit_with:adom_distty"

    (* tries to prove that a condition holds
     * soundness: if sat e t returns true, all states in gamma(t) satisfy e *)
    let sat (e: expr) (t: t): bool = failwith "todo:sat"

    (* Dimensions management *)
    let dim_add (dn: string) (t: t): t =
      { t_samp = SS.add dn              t.t_samp;
        t_dist = SM.add dn (DBot, VBot) t.t_dist;
        t_aux = top.t_aux; }
    let dim_rem (dn: string) (t: t): t =
      { t_samp = SS.remove dn t.t_samp;
        t_dist = SM.remove dn t.t_dist;
        t_aux = top.t_aux; }
    (* let dim_rem_set (dns: SS.t) (t: t): t = failwith "todo:dim_rem_set" *)
    let dim_mem (dn: string) (t: t): bool = SS.mem dn t.t_samp
    let dim_project_out (dn: string) (t: t): t = dim_rem dn t
    let dims_get (t: t): SS.t option = Some t.t_samp

    (* ad-hoc function *)
    let set_aux_distty (a: aux_ty) (t: t) = { t with t_aux = a; }
    let bound_var_apron s t =
      failwith "adom_distty.ml: bound_var_apron must not be called!"

    (* checks whether a domain-specific relationship holds
     * between two abstract elements *)
    let is_related_dist_lat dl1 dl2 = 
      match dl1, dl2 with
      | DSome dk1, DSome dk2 -> IU.dist_kind_support_subseteq dk2 dk1
      | _ -> false

    let is_related_vtyp_lat vl1 vl2 = 
      match vl1, vl2 with
      | VSome v1, VSome v2 -> v1 = v2
      | _ -> false

    let is_related t1 t2 =
      let t1_minus_t2 = SS.diff t1.t_samp t2.t_samp in
      let t2_minus_t1 = SS.diff t2.t_samp t1.t_samp in
      let is_simple_dist s = 
        try 
          match fst (SM.find s t1.t_dist) with
          | DSome (Subsample _) -> true
          | _ -> false
        with Not_found -> false
      in
      SS.is_empty t2_minus_t1
      && SS.for_all is_simple_dist t1_minus_t2
      && SM.for_all (fun s ((dl1,vl1),(dl2,vl2)) -> 
                       is_related_dist_lat dl1 dl2 
                       && is_related_vtyp_lat vl1 vl2)
                    (merge_smap t1.t_dist t2.t_dist)

    let range_info e t =
      failwith "todo:range_info:adom_distty"
  end: ABST_DOMAIN_NB_D)
