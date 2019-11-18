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
 ** adom_fib.ml: prototype implementation of a fibered domain functor *)
open Lib
open Ir_sig
open Irty_sig
open Adom_sig

module AM = Ai_make
module IU = Ir_util
module ITU = Irty_util
module AZ = Adom_zone
module NM = NameManager
module IC = Ir_cast

type maymust = Must | May

(** Helper functions : naming dimensions *)
let dim_sample_low_make (mm: maymust) (s: string): string =
  match mm with
  | Must -> Printf.sprintf "LOW(MUST(%s))" s
  | May  -> Printf.sprintf "LOW(MAY(%s))" s
let dim_sample_high_make (mm:maymust) (s: string): string =
  match mm with
  | Must -> Printf.sprintf "HIGH(MUST(%s))" s
  | May  -> Printf.sprintf "HIGH(MAY(%s))" s
let dim_tens_size_make (s: string): string =
  Printf.sprintf "TENS_SZ(%s)" s
let dim_ptrn_unmake (ptrn: string): string -> string option =
  let lp = String.length ptrn in
  fun (s: string): string option ->
    let len = String.length s in
    if (len > lp + 2 && String.sub s 0 lp = ptrn
                     && String.get s lp = '('
                     && String.get s (len-1) = ')')
    then Some (String.sub s (lp+1) (len-1))
    else None
let dim_ptrn_unmake_cond (f: string -> bool) (ptrn: string) (s: string): bool =
  match dim_ptrn_unmake ptrn s with
  | None -> false
  | Some v -> try f v with Not_found -> false

(* wy: these funcs will be removed if we decide to use zone only. *)
let is_with_1d_idx (s: string) =
  not (String.sub s 0 2 = "__")
let get_name_with_1d_idx (ss: SS.t) =
  SS.filter (fun s ->     (is_with_1d_idx s)) ss
let get_name_with_2d_idx (ss: SS.t) =
  SS.filter (fun s -> not (is_with_1d_idx s)) ss

(** Module constructor *)
module Make = functor (D_num: ABST_DOMAIN_ZONE_NB) (D_typ: ABST_DOMAIN_NB_D) ->
  (struct
    let module_name =
      Printf.sprintf "FibFull(%s,%s)" D_num.module_name D_typ.module_name

    (* t is the type for abstract states:
     * A type (vtyp, sampidx, sampvar, u_num, u_typ) describes the set
     * of states where
     *     the variables in the domain of vtyp have types specified
     *         by vtyp,
     *     the strings in sampidx denote string constructors used
     *         for indexed sampled random variables,
     *     the strings in sampvar denote strings variables used
     *         for sampled random variables,
     *     u_num describes numerical constraints satisfied by
     *         the state, and
     *     v_typ describes information about the distributions of
     *         sampled random variables indexed or non-indexed.
     *
     * Invariants on dim(t_u_num):
     *  - for each name s with type (Vt_num nt), there is a dimension
     *    in t_u_num called:         s
     *  - for each name s with type (t_tens ts), there is a dimension
     *    in t_u_num called:         dim_tens_size_make s
     *  - for each name s in t_sampidx.m_may, there are two dimensions
     *    in t_u_num called:
     *          dim_sample_low_make  May s
     *          dim_sample_high_make May s
     *  - for each name s in t_sampidx.m_must, there are two dimensions
     *    in t_u_num called:
     *          dim_sample_low_make  Must s
     *          dim_sample_high_make Must s
     *
     * Invariant on dim(t_u_typ):
     *  - dim(t_u_typ) = t_sampidx.m_must \cup t_sampvar.m_must
     *)
    type mm =
        { m_may:  SS.t ;
          m_must: SS.t }
    type t =
        { (* types of variables *)
          t_vtyp:     vtyp SM.t;
          (* component "a": string constructors, name of sampled random vars *)
          t_sampidx:  mm ;
          (* component "b": name of sampled random vars *)
          t_sampvar:  mm ;
          (* underlying domain abstract value *)
          t_u_num:    D_num.t ;
          (* underlying domain for distribution type *)
          t_u_typ:    D_typ.t ;
          (* broadcast information *)
          t_bcast:    broadcast_info }

    (********************)
    (* Helper functions *)
    (********************)
    let get_typ t typ =
      SM.fold
        (fun v0 t0 acc -> if ITU.vtyp_leq t0 typ then SS.add v0 acc else acc)
        t.t_vtyp SS.empty

    (************)
    (* printing *)
    (************)
    let buf_t_vtyp_tens (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "{ ";
      let iter_fun v t =
        match t with
        | Vt_tens _ -> Printf.bprintf buf "%s:%s; " v (ITU.vtyp_to_string t)
        | _ -> () in
      SM.iter iter_fun t.t_vtyp;
      Printf.bprintf buf "}"

    let buf_t_vtyp_plate (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "{ ";
      let iter_fun v0 t0 =
        match t0 with
        | Vt_plate _ -> Printf.bprintf buf "%s:%s; " v0 (ITU.vtyp_to_string t0)
        | _ -> () in
      SM.iter iter_fun t.t_vtyp;
      Printf.bprintf buf "}"

    let buf_t_vtyp_range (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "{ ";
      let iter_fun v0 t0 =
        match t0 with
        | Vt_range _ -> Printf.bprintf buf "%s:%s; " v0 (ITU.vtyp_to_string t0)
        | _ -> () in
      SM.iter iter_fun t.t_vtyp;
      Printf.bprintf buf "}"

    let buf_t_vtyp_distr (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "{ ";
      let iter_fun v0 t0 =
        match t0 with
        | Vt_distr _ -> Printf.bprintf buf "%s:%s; " v0 (ITU.vtyp_to_string t0)
        | _ -> () in
      SM.iter iter_fun t.t_vtyp;
      Printf.bprintf buf "}"

    let buf_t_vtyp_fun (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "{ ";
      let iter_fun v0 t0 =
        match t0 with
        | Vt_fun _ -> Printf.bprintf buf "%s:%s; " v0 (ITU.vtyp_to_string t0)
        | _ -> () in
      SM.iter iter_fun t.t_vtyp;
      Printf.bprintf buf "}"

    let buf_t (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "[ATO] %a, %a, %a\n"
        buf_ss (get_typ t Vt_nil)
        buf_ss (get_typ t (Vt_num NT_int))
        buf_ss (get_typ t (Vt_num NT_real));
      Printf.bprintf buf "[PLA] %a\n" buf_t_vtyp_plate t;
      Printf.bprintf buf "[RNG] %a\n" buf_t_vtyp_range t;
      Printf.bprintf buf "[DST] %a\n" buf_t_vtyp_distr t;
      Printf.bprintf buf "[FUN] %a\n" buf_t_vtyp_fun t;
      Printf.bprintf buf "[TEN] %a\n" buf_t_vtyp_tens t;
      Printf.bprintf buf "[RVS] %a, %a, %a, %a\n"
        buf_ss t.t_sampidx.m_must
        buf_ss t.t_sampvar.m_must
        buf_ss t.t_sampidx.m_may
        buf_ss t.t_sampvar.m_may;
      Printf.bprintf buf "[TYP] %a\n" D_typ.buf_t t.t_u_typ;
      Printf.bprintf buf "[BCT] %s\n" (ITU.broadcast_info_to_string t.t_bcast);
      Printf.bprintf buf "[APR]\n%a\n" D_num.buf_t t.t_u_num
    let to_string = buf_to_string buf_t
    let pp = buf_to_channel buf_t

    (******************)
    (* lattice values *)
    (******************)
    let mm_top: mm =
      { m_may  = SS.empty;
        m_must = SS.empty }
    let top =
      { t_vtyp     = SM.empty ;
        t_sampidx  = mm_top ;
        t_sampvar  = mm_top ;
        t_u_num    = D_num.top ;
        t_u_typ    = D_typ.top ;
        t_bcast    = [] }
    let is_bot (t: t) = D_num.is_bot t.t_u_num || D_typ.is_bot t.t_u_typ

    let tens_id_fn_ty = FT_tens_resize([],[],[])
    let tens_identity_fns =
      [("torch.exp",           tens_id_fn_ty);
       ("torch.sigmoid",       tens_id_fn_ty);
       ("F.relu",              tens_id_fn_ty);
       ("F.softplus",          tens_id_fn_ty);
       ("nn.Parameter",        tens_id_fn_ty);
       ("torch.Tensor.detach", tens_id_fn_ty);
       ("torch.Tensor.scatter_add_", tens_id_fn_ty);]
    (* https://pytorch.org/docs/stable/torch.html#torch.exp
     * https://pytorch.org/docs/stable/torch.html#torch.sigmoid
     * https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#relu
     * https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softplus
     * https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter
     * https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach *)
    let tens_identity_cls = ["nn.ReLU"; "nn.Softplus"; "nn.Sigmoid"; "nn.Tanh"; "nn.Softmax"]
    let tens_resize_cls = ["nn.Linear"; "nn.LSTMCell"; "nn.RNN"]
    (* https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
     * https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell 
     * https://pytorch.org/docs/stable/nn.html#torch.nn.RNN *)
    let init_t: t =
      { top with
        t_vtyp = List.fold_left 
                   (fun m (k,ft) -> SM.add k (Vt_fun ft) m) 
                   top.t_vtyp 
                   tens_identity_fns }

    (*************************)
    (* sanity check (helper) *)
    (*************************)
    (* tries to verify the above invariant *)
    let sanity_check (t: t): unit =
      let fail msg =
        Printf.eprintf "Invalid abstract state: %s\n%a\n" msg pp t;
        failwith "Corrupted analysis state; Analysis quits..." in
      match D_num.dims_get t.t_u_num with
      | None ->
          Printf.eprintf "sanity_check: underlying dom does not give dimensions"
      | Some s ->
          let r = ref s in
          let check_dim d =
            if SS.mem d !r then r := SS.remove d !r
            else fail (Printf.sprintf "dimension %s missing" d) in
          let t_t_int  = get_typ t (Vt_num NT_int) in
          let t_t_real = get_typ t (Vt_num NT_real) in
          let t_t_tens = get_typ t (Vt_tens None) in
          SS.iter (fun s -> check_dim s) t_t_int;
          SS.iter (fun s -> check_dim s) t_t_real;
          SS.iter (fun s -> check_dim (dim_tens_size_make s)) t_t_tens;
          SS.iter (fun s -> check_dim (dim_sample_low_make  Must s))
            t.t_sampidx.m_must;
          SS.iter (fun s -> check_dim (dim_sample_high_make Must s))
            t.t_sampidx.m_must;
          SS.iter (fun s -> check_dim (dim_sample_low_make  May  s))
            t.t_sampidx.m_may;
          SS.iter (fun s -> check_dim (dim_sample_high_make May  s))
            t.t_sampidx.m_may;
          if !r <> SS.empty then
            fail (Printf.sprintf "unbound dimensions, %s" (ss_to_string !r))

    (********************************)
    (* Managing dimensions (helper) *)
    (********************************)
    (* Function list:
     * - mm_update, mm_add, mm_rem
     * - dim_{num,tens}_{add,rem,upd}
     * - dim_{sampidx,sampvar}_{add,rem,rem_set}
     * - dim_{fun}_{add,rem,upd} *)

    (* Update a may-must element *)
    let mm_update (u: maymust) (f: SS.t -> SS.t) (m: mm): mm =
      match u with
      | May  -> { m with m_may  = f m.m_may }
      | Must -> { m with m_must = f m.m_must }
    let mm_add (u: maymust) (s: string): mm -> mm = mm_update u (SS.add s)
    let mm_rem (u: maymust) (s: string): mm -> mm = mm_update u (SS.remove s)

    (* Management of dimensions for int/real variables 
     * - dim_num_upd assumes that t_vtyp(x) = Vt_num _. 
     * - dim_num_add assumes that t_vtyp(x) is undefined *)
    let dim_num_upd (x: string) (nt: num_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_num nt) t.t_vtyp }
    let dim_num_add (x: string) (nt: num_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_num nt) t.t_vtyp;
        t_u_num = D_num.dim_add x t.t_u_num }
    let dim_num_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp ;
        t_u_num = D_num.dim_rem x t.t_u_num }

    (* Management of dimensions for plate variables
     * - dim_plate_upd assumes that t_vtyp(x) = Vt_plate _. 
     * - dim_plate_add assumes that t_vtyp(x) is undefined *)
    let dim_plate_upd (x: string) (pt: plate_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_plate pt) t.t_vtyp }
    let dim_plate_add (x: string) (pt: plate_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_plate pt) t.t_vtyp }
    let dim_plate_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp }

    (* Management of dimensions for range variables
     * - dim_range_upd assumes that t_vtyp(x) = Vt_range _. 
     * - dim_range_add assumes that t_vtyp(x) is undefined *)
    let dim_range_upd (x: string) (rt: range_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_range rt) t.t_vtyp }
    let dim_range_add (x: string) (rt: range_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_range rt) t.t_vtyp }
    let dim_range_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp }

    (* Management of dimensions for distr variables
     * - dim_distr_upd assumes that t_vtyp(x) = Vt_distr _. 
     * - dim_distr_add assumes that t_vtyp(x) is undefined *)
    let dim_distr_upd (x: string) (dt: distr_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_distr dt) t.t_vtyp }
    let dim_distr_add (x: string) (dt: distr_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_distr dt) t.t_vtyp }
    let dim_distr_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp }

    (* Management of dimensions for fun variables 
     * - dim_fun_upd assumes that t_vtyp(x) = Vt_fun _. 
     * - dim_fun_add assumes that t_vtyp(x) is undefined *)
    let dim_fun_upd (x: string) (ft: fun_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_fun ft) t.t_vtyp }
    let dim_fun_add (x: string) (ft: fun_ty) (t: t): t =
      { t with
        t_vtyp  = SM.add x (Vt_fun ft) t.t_vtyp }
    let dim_fun_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp }

    (* Management of dimensions for nil variables 
     * - dim_nil_add assumes that t_vtyp(x) is undefined *)
    let dim_nil_add (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.add x Vt_nil t.t_vtyp }
    let dim_nil_rem (x: string) (t: t): t =
      { t with
        t_vtyp  = SM.remove x t.t_vtyp }

    (* Management of dimensions for tensor sizes *)
    let dim_tens_upd (s: string) (ts: tensor_size_ty) (t: t): t =
      { t with
        t_vtyp = SM.add s (Vt_tens ts) t.t_vtyp }
    let dim_tens_add (s: string) (ts: tensor_size_ty) (t: t): t =
      let dsize = dim_tens_size_make s in
      { t with
        t_vtyp  = SM.add s (Vt_tens ts) t.t_vtyp ;
        t_u_num = D_num.dim_add dsize t.t_u_num }
    let dim_tens_rem (s: string) (t: t): t =
      let dsize = dim_tens_size_make s in
      { t with
        t_vtyp  = SM.remove s t.t_vtyp ;
        t_u_num = D_num.dim_rem dsize t.t_u_num }

    (* Management of dimensions for indexed sample statements *)
    let dim_sampidx_add (mm: maymust) (s: string) (t: t): t * string * string =
      (* wy: will be simplified *)
      let dlow  = dim_sample_low_make mm s
      and dhigh = dim_sample_high_make mm s in
      let t_u_num = D_num.dim_add dlow (D_num.dim_add dhigh t.t_u_num) in
      let t_u_typ, t_u_num =
        match mm with
        | Must -> D_typ.dim_add s t.t_u_typ,
                  D_num.zone_dim_add s t_u_num
        | May  -> t.t_u_typ,
                  t_u_num in
      { t with
        t_sampidx = mm_add mm s t.t_sampidx ;
        t_u_typ   = t_u_typ ;
        t_u_num   = t_u_num }, dlow, dhigh
    let dim_sampidx_rem (mm: maymust) (s: string) (t: t): t =
      (* wy: will be simplified *)
      let dlow  = dim_sample_low_make mm s
      and dhigh = dim_sample_high_make mm s in
      let t_u_num = D_num.dim_rem dlow (D_num.dim_rem dhigh t.t_u_num) in
      let t_u_typ, t_u_num = 
        match mm with
        | Must -> D_typ.dim_rem s t.t_u_typ,
                  D_num.zone_dim_rem s t_u_num
        | May  -> t.t_u_typ,
                  t_u_num in
      { t with
        t_sampidx = mm_rem mm s t.t_sampidx ;
        t_u_typ   = t_u_typ ;
        t_u_num   = t_u_num }

    (* Management of dimensions for non-indexed sample statements *)
    let dim_sampvar_add (mm: maymust) (s: string) (t: t): t =
      let t_u_typ_new =
        match mm with
        | Must -> D_typ.dim_add s t.t_u_typ
        | May  -> t.t_u_typ in
      { t with
        t_sampvar = mm_add mm s t.t_sampvar ;
        t_u_typ   = t_u_typ_new }
    let dim_sampvar_rem (mm: maymust) (s: string) (t: t): t =
      let t_u_typ_new =
        match mm with
        | Must -> D_typ.dim_rem s t.t_u_typ
        | May  -> t.t_u_typ in
      { t with
        t_sampvar = mm_rem mm s t.t_sampvar ;
        t_u_typ   = t_u_typ_new }

    (* Addition/Removal of sets of variables *)
    let dim_sampidx_add_set (mm: maymust) (s: SS.t) (t: t): t =
      (* wy: can be simplified if we decide to use zone only. *)
      let dim_sampidx_add_new mm s t =
        let (t_new, _, _) = dim_sampidx_add mm s t in
        t_new in
      SS.fold (dim_sampidx_add_new mm) s t
    let dim_sampvar_add_set (mm: maymust) (s: SS.t) (t: t): t =
      SS.fold (dim_sampvar_rem mm) s t
    let dim_sampidx_rem_set (mm: maymust) (s: SS.t) (t: t): t =
      SS.fold (dim_sampidx_rem mm) s t
    let dim_sampvar_rem_set (mm: maymust) (s: SS.t) (t: t): t =
      SS.fold (dim_sampvar_rem mm) s t

    (**********************)
    (* lattice operations *)
    (**********************)
    (* Helper functions: make environment consistent *)
    let t_combine_may (t0: t) (t1: t): t =
      (* extend t_samp{idx,var} *)
      let sampidx_diff = SS.diff t1.t_sampidx.m_may t0.t_sampidx.m_may in
      let sampvar_diff = SS.diff t1.t_sampvar.m_may t0.t_sampvar.m_may in
      let t0 = dim_sampidx_add_set May sampidx_diff t0 in
      let t0 = dim_sampvar_add_set May sampvar_diff t0 in
      (* handle t_u_num accordingly *)
      (* wy: can be simplified if we decide to use zone only. *)
      let dim_sample_low_diff  =
        SS.map (fun s -> dim_sample_low_make  May s) sampidx_diff in
      let dim_sample_high_diff =
        SS.map (fun s -> dim_sample_high_make May s) sampidx_diff in
      let t0 =
        SS.fold
          (fun s acc ->
            { acc with
              t_u_num = D_num.eval (Assn (s, Num(Int(0))))
                       (D_num.eval (Assn (s, Num(Int(1)))) acc.t_u_num) }
          ) (SS.union dim_sample_low_diff dim_sample_high_diff) t0 in
      t0

    let t_combine_must (t0: t) (t1: t): t =
      (* shrink t_samp{idx,var} *)
      let sampidx_diff = SS.diff t0.t_sampidx.m_must t1.t_sampidx.m_must in
      let sampvar_diff = SS.diff t0.t_sampvar.m_must t1.t_sampvar.m_must in
      let t0 = dim_sampidx_rem_set Must sampidx_diff t0 in
      let t0 = dim_sampvar_rem_set Must sampvar_diff t0 in
      t0

    let t_diff_dim (t0: t) (t1: t): t =
      SM.fold
        (fun x vt0 acc ->
          let ovt1 = try Some (SM.find x t1.t_vtyp) with Not_found -> None in
          match vt0, ovt1 with
          | Vt_nil        , Some Vt_nil           -> acc
          | Vt_nil        , _                     -> dim_nil_rem x acc
          | Vt_num nt0    , Some (Vt_num nt1)     ->
              let nt = ITU.num_ty_join nt0 nt1 in
              if nt = nt0 then acc else dim_num_upd x nt acc
          | Vt_num _      , _                     -> dim_num_rem x acc
          | Vt_plate pt0  , Some (Vt_plate pt1)   -> 
              let pt = ITU.plate_ty_join pt0 pt1 in
              if pt = pt0 then acc else dim_plate_upd x pt acc
          | Vt_plate _    , _                     -> dim_plate_rem x acc
          | Vt_range rt0  , Some (Vt_range rt1)   -> 
              let rt = ITU.range_ty_join rt0 rt1 in
              if rt = rt0 then acc else dim_range_upd x rt acc
          | Vt_range _    , _                     -> dim_range_rem x acc
          | Vt_distr dt0  , Some (Vt_distr dt1)   -> 
              let dt = ITU.distr_ty_join dt0 dt1 in
              if dt = dt0 then acc else dim_distr_upd x dt acc
          | Vt_distr _    , _                     -> dim_distr_rem x acc
          | Vt_fun ft0    , Some (Vt_fun ft1)     -> 
              let ft = ITU.fun_ty_join ft0 ft1 in
              if ft = ft0 then acc else dim_fun_upd x ft acc
          | Vt_fun _      , _                     -> dim_fun_rem x acc
          | Vt_tens ts0   , Some (Vt_tens ts1)    ->
              let ts = ITU.tensor_size_ty_join ts0 ts1 in
              if ts = ts0 then acc else dim_tens_upd x ts acc
          | Vt_tens _     , _                     -> dim_tens_rem x acc
        ) t0.t_vtyp t0 

    (* Helper function *)
    let bin_combine (f: D_num.t -> D_num.t -> D_num.t)
                    (g: D_typ.t -> D_typ.t -> D_typ.t)
                    (t0: t) (t1: t): t =
      let _ =
        if !AM.debug then
          (Printf.printf "t0_in:\n%a\n" pp t0;
           Printf.printf "t1_in:\n%a\n" pp t1) in
      (* extend may  set: t_samp{var,idx}.m_may, t_u_num *)
      let t0 = t_combine_may  t0 t1 in
      let t1 = t_combine_may  t1 t0 in
      (* shrink must set: t_samp{var,idx}.m_must, t_u_num, t_u_typ *)
      let t0 = t_combine_must t0 t1 in
      let t1 = t_combine_must t1 t0 in
      (* shrink dim: t_vtyp, t_u_num *)
      let t0 = t_diff_dim t0 t1 in
      let t1 = t_diff_dim t1 t0 in
      (* return *)
      let _ =
        if !AM.debug then
          (Printf.printf "t0:\n%a\n" pp t0;
           Printf.printf "t1:\n%a\n" pp t1) in
      { t0 with
        t_u_num = f t0.t_u_num t1.t_u_num ;
        t_u_typ = g t0.t_u_typ t1.t_u_typ }

    (* Main functions *)
    let join: t -> t -> t =
      bin_combine D_num.join D_typ.join
    let widen (thr: expr list): t -> t -> t =
      bin_combine (D_num.widen thr) (D_typ.widen thr)
    let leq (t0: t) (t1: t): bool =
      let leq_vtyp vt0 vt1 =
        let module M = struct exception Stop end in
        try (* check if all constraints in vt1 hold in vt0 *)
          SM.iter
            (fun x tp1 ->
              try
                let tp0 = SM.find x vt0 in
                if not (ITU.vtyp_leq tp0 tp1) then raise M.Stop
              with Not_found -> raise M.Stop
            ) vt1;
          true
        with M.Stop -> false in
      leq_vtyp t0.t_vtyp t1.t_vtyp
        && SS.subset t1.t_sampidx.m_must t0.t_sampidx.m_must
        && SS.subset t1.t_sampvar.m_must t0.t_sampvar.m_must
        && SS.subset t0.t_sampidx.m_may t1.t_sampidx.m_may
        && SS.subset t0.t_sampvar.m_may t1.t_sampvar.m_may
        && D_num.leq (t_diff_dim t0 t1).t_u_num t1.t_u_num
        && D_typ.leq (t_diff_dim t0 t1).t_u_typ t1.t_u_typ

    (********)
    (* eval *)
    (********)
    (* Helper functions: extraction of type information.
     * Function list:
     * - get_exp_ty
     * - is_{int,real,tensor}_exp
     * - eval_exp *)

    let get_name_ty t x =
      try
        match SM.find x t.t_vtyp with
        | Vt_nil      -> ET_nil
        | Vt_num nt   -> ET_num nt
        | Vt_plate pt -> ET_plate pt
        | Vt_range rt -> ET_range rt
        | Vt_distr dt -> ET_distr dt
        | Vt_fun ft   -> ET_fun ft
        | Vt_tens ts  -> ET_tensor ts
      with
      | Not_found ->
         let t_sampidx = SS.union t.t_sampidx.m_must t.t_sampidx.m_may in
         (* wy: can be simplified if we decide to use zone only. *)
         if dim_ptrn_unmake_cond    (fun s -> SS.mem s t_sampidx) "LOW"  x
            || dim_ptrn_unmake_cond (fun s -> SS.mem s t_sampidx) "HIGH" x
            || dim_ptrn_unmake_cond (fun s -> match SM.find s t.t_vtyp with
                                               | Vt_tens _ -> true
                                               | _ -> false)
                                    "TENS_SZ" x then
            ET_num NT_int
         else ET_unknown
    let get_exp_ty    t e = ITU.get_exp_ty    (get_name_ty t) e
    let is_int_exp    t e = ITU.is_int_exp    (get_name_ty t) e
    let is_real_exp   t e = ITU.is_real_exp   (get_name_ty t) e
    let is_tensor_exp t e = ITU.is_tensor_exp (get_name_ty t) e
    let is_fun_exp    t e = ITU.is_fun_exp    (get_name_ty t) e
    let eval_exp      t e = IU.simplify_exp   (get_exp_ty t)  e

    (* Helper function: management of dimensions for type changes
     * - when the new type is "None", discards type information *)
    let var_becomes_typ (x: string) (nt: vtyp option) (t: t): t =
      let ot = try Some (SM.find x t.t_vtyp) with Not_found -> None in
      let remove t0 = function
        | None              -> t0
        | Some Vt_nil       -> dim_nil_rem x t0
        | Some (Vt_num _)   -> dim_num_rem x t0
        | Some (Vt_plate _) -> dim_plate_rem x t0
        | Some (Vt_range _) -> dim_range_rem x t0
        | Some (Vt_distr _) -> dim_distr_rem x t0
        | Some (Vt_fun _)   -> dim_fun_rem x t0
        | Some (Vt_tens _)  -> dim_tens_rem x t0 in
      let add t0 = function
        | None                -> t0
        | Some Vt_nil         -> dim_nil_add x t0
        | Some (Vt_num nt1)   -> dim_num_add x nt1 t0
        | Some (Vt_plate pt1) -> dim_plate_add x pt1 t0
        | Some (Vt_range rt1) -> dim_range_add x rt1 t0
        | Some (Vt_distr dt1) -> dim_distr_add x dt1 t0
        | Some (Vt_fun ft1)   -> dim_fun_add x ft1 t0
        | Some (Vt_tens ts1)  -> dim_tens_add x ts1 t0 in
      match ot, nt with
      | Some Vt_nil        , Some Vt_nil         -> t
      | Some (Vt_num nt0)  , Some (Vt_num nt1)   -> if nt0 = nt1 then t
                                                    else dim_num_upd x nt1 t
      | Some (Vt_plate pt0), Some (Vt_plate pt1) -> if pt0 = pt1 then t
                                                    else dim_plate_upd x pt1 t
      | Some (Vt_range rt0), Some (Vt_range rt1) -> if rt0 = rt1 then t
                                                    else dim_range_upd x rt1 t
      | Some (Vt_distr dt0), Some (Vt_distr dt1) -> if dt0 = dt1 then t
                                                    else dim_distr_upd x dt1 t
      | Some (Vt_fun ft0)  , Some (Vt_fun ft1)   -> if ft0 = ft1 then t
                                                    else dim_fun_upd x ft1 t
      | Some (Vt_tens ts0) , Some (Vt_tens ts1)  -> if ts0 = ts1 then t
                                                    else dim_tens_upd x ts1 t
      | _                  , _                   -> add (remove t ot) nt

    (* Helper functions: implementing parts of transfer functions for tensors
     * Note: eval func must call one of the following.
     * Function list:
     * - var_becomes_nil
     * - var_assign_num
     * - var_becomes_plate
     * - var_becomes_range
     * - var_becomes_distr
     * - var_becomes_fun
     * - var_becomes_other_havoc
     * - var_becomes_tens_{size,var,havoc} *)

    (* transfer function for x:=e where e evaluates to None *)
    let var_becomes_nil (x: string) (t: t): t =
      var_becomes_typ x (Some Vt_nil) t

    (* write an expression into a numeric variable:
     * - a is None when we over-approximate the result as "any value" of
     *   the corresponding type
     * - v may be either (Vt_num NT_int) or (Vt_num NT_real) *)
    let var_assign_num (x: string) (v: num_ty) (a: acmd option) (t: t): t =
      let t = var_becomes_typ x (Some (Vt_num v)) t in
      match a with
      | None    -> { t with t_u_num = D_num.dim_project_out x t.t_u_num }
      | Some ac -> { t with t_u_num = D_num.eval ac t.t_u_num }

    (* transfer function for x:=e when e has a plate type *)
    let var_becomes_plate (x: string) (pt: plate_ty) (t: t): t =
      var_becomes_typ x (Some (Vt_plate pt)) t 

    (* transfer function for x:=e when e has a range type *)
    let var_becomes_range (x: string) (rt: range_ty) (t: t): t =
      var_becomes_typ x (Some (Vt_range rt)) t 

    (* transfer function for x:=e when e has a distr type *)
    let var_becomes_distr (x: string) (dt: distr_ty) (t: t): t =
      var_becomes_typ x (Some (Vt_distr dt)) t 

    (* transfer function for x:=e when e has a function type *)
    let var_becomes_fun (x: string) (ft: fun_ty) (t: t): t =
      var_becomes_typ x (Some (Vt_fun ft)) t

    (* transfer function for x:=? when ? has unknown type *)
    let var_becomes_other_havoc (x: string) (t: t): t =
      var_becomes_typ x None t

    (* convert a varable x to a tensor.
     * set its tensor shape to 'ts'.
     * set its first dim to 'size' when 'fst_dim_opt = Some size', or
     * forget its first dim when 'fst_dim_opt = None'.
     * here 'fst_dim_opt' is computed from 'ts' when 'fst_dim = (true, _)', or
     * 'fst_dim_opt' is set to 'eopt' when 'fst_dim = (false, eopt).' *)
    let var_becomes_tens_size (x: string) (ts: tensor_size_ty)
          (fst_dim: bool * expr option) (t: t): t =
      let get_fst_dim: tensor_size_ty -> expr option = function
        | Some (Some n :: _) -> Some (Num (Int n)) 
        | _ -> None in
      let x_size = dim_tens_size_make x in
      let t = var_becomes_typ x (Some (Vt_tens ts)) t in
      let fst_dim_opt =
        if fst fst_dim then get_fst_dim ts
        else                snd fst_dim in
      match fst_dim_opt with
      | None      ->
          { t with t_u_num = D_num.dim_project_out x_size t.t_u_num }
      | Some size ->
          { t with t_u_num = D_num.eval (Assn (x_size, size)) t.t_u_num }

    (* transfer function for x:=y when x and y are known to be
     * tensors and they are different. *)
    let var_becomes_tens_var (x: string) (y: string) (ts: tensor_size_ty) (t: t): t =
      let y_size = dim_tens_size_make y in
      var_becomes_tens_size x ts (false, (Some (Name y_size))) t

    (* transfer function for x:=? when ? is known to be a tensor *)
    let var_becomes_tens_havoc (x: string) (t: t): t =
      var_becomes_tens_size x None (false, None) t

    (* Helper functions: update t_u_{num,typ}.
     * Function list:
     * - update_name_sets
     * - update_distty *)

    let is_allocated_str (s: string) (t: t): bool =
      SS.mem s t.t_sampvar.m_must

    let is_allocated_strfmt (s: string) (e: expr) (t: t): bool =
      let dhigh = dim_sample_high_make Must s in
      let dlow = dim_sample_low_make Must s in
      SS.mem s t.t_sampidx.m_must
      && D_num.sat (Comp (Lt, e, Name dhigh)) t.t_u_num
      && D_num.sat (Comp (LtE, Name dlow, e)) t.t_u_num

    let is_allocated_strfmt_zone (s: string) (idx: bnd_expr list) (t: t): bool =
      SS.mem s t.t_sampidx.m_must
      && D_num.zone_include idx s t.t_u_num

    let update_name_sets (t: t): expr -> t = function
      | Str s ->
          if is_allocated_str s t then
            raise (Must_error "update_name_sets: same name for >1 random vars");
          let t = dim_sampvar_add Must s t in
          let t = dim_sampvar_add May  s t in
          t
      (* wy: this pattern matching can be removed if we decide to use zone only. *)
      | StrFmt (s, [e]) ->
         (* update must set *)
         if is_allocated_strfmt s e t then
           raise (Must_error
                    "update_name_sets: same indexed name for >1 random vars");
         let t =
           let ep1 = BOp (Add, e, Num (Int 1)) in
           if SS.mem s t.t_sampidx.m_must then
             (* dimensions do already exist;
              * we try to extend the range corresponding to the idx sample *)
             let dhigh = dim_sample_high_make Must s in
             if D_num.sat (Comp (Eq, Name dhigh, e)) t.t_u_num then
               (* case where the range can successfully be extended at the top *)
               let u = D_num.eval (Assn (dhigh, ep1)) t.t_u_num in
               { t with t_u_num = u }
             else
               let dlow = dim_sample_low_make Must s in
               if D_num.sat (Comp (Eq, Name dlow, ep1)) t.t_u_num then
                 (* case where the range can successfully be extended
                  * at the bottom *)
                 let u = D_num.eval (Assn (dlow, e)) t.t_u_num in
                 { t with t_u_num = u }
               else
                 (* case where we may not be able to extend the range
                  * we return t, which indicates we do approximation *)
                 t
           else
             (* dimensions do not exist;
              * add both, and the constraint low=e /\ high=e+1 *)
             let t, dlow, dhigh = dim_sampidx_add Must s t in
             let u = D_num.eval (Assume (Comp (Eq, Name dlow, e))) t.t_u_num in
             let u = D_num.eval (Assume (Comp (Eq, Name dhigh, ep1))) u in
             { t with t_u_num = u } in

         (* update may set *)
         let t =
           let ep1 = BOp (Add, e, Num (Int 1)) in
           if SS.mem s t.t_sampidx.m_may then
             (* dimensions do already exist;
              * we try to extend the range corresponding to the idx sample *)
             let dhigh = dim_sample_high_make May s in
             if D_num.sat (Comp (GtE, Name dhigh, e)) t.t_u_num then
               (* case where the range can successfully be extended at the top *)
               let u = D_num.eval (Assn (dhigh, ep1)) t.t_u_num in
               { t with t_u_num = u }
             else
               let dlow = dim_sample_low_make Must s in
               if D_num.sat (Comp (LtE, Name dlow, ep1)) t.t_u_num then
                 (* case where the range can successfully be extended
                  * at the bottom *)
                 let u = D_num.eval (Assn (dlow, e)) t.t_u_num in
                 { t with t_u_num = u }
               else
                 (* case where we may not be able to extend the range
                  * we return t, which indicates we do approximation *)
                 t
           else
             (* dimensions do not exist;
              * add both, and the constraint low=e /\ high=e+1 *)
             let t, dlow, dhigh = dim_sampidx_add May s t in
             let u = D_num.eval (Assume (Comp (Eq, Name dlow, e))) t.t_u_num in
             let u = D_num.eval (Assume (Comp (Eq, Name dhigh, ep1))) u in
             { t with t_u_num = u } in
         t
      | StrFmt (s, args) ->
         let idx = List.map AZ.expr_to_bnd_expr args in
         if is_allocated_strfmt_zone s idx t then
           raise (Must_error
                    "update_name_sets: same indexed name for >1 random vars");
         let t,_,_ = 
           if SS.mem s t.t_sampidx.m_must
           then t, "", ""
           else dim_sampidx_add Must s t in
         let t,_,_ =
           if SS.mem s t.t_sampidx.m_may
           then t, "", ""
           else dim_sampidx_add May  s t in
         { t with
           t_u_num = D_num.zone_add_cell s idx t.t_u_num }
      | _ -> t

    let update_distty (t: t): acmd -> t = function
      | Sample(x, _, _, _, _) as ac ->
          let x_vtyp = try Some (SM.find x t.t_vtyp) with Not_found -> None in
          (* first pass x_vtyp to t, and then do eval to update t *)
          let t_u_typ = D_typ.set_aux_distty x_vtyp t.t_u_typ in
          let t_u_typ = D_typ.eval ac t_u_typ in
          { t with t_u_typ = t_u_typ }
      | _ -> failwith "update_distty: invalid acmd"

    (* Helper functions: extract range info (int^3 opt) from args of range of plate.
     * Function list:
     * - _range_info_from_{range,plate} *)
    let _range_info_from_range: int option list -> (int * int * int) option = function
      | [Some u]                 -> Some (0, u, 1)
      | [Some l; Some u]         -> Some (l, u, 1)
      | [Some l; Some u; Some s] -> Some (l, u, s)
      | _                        -> None
    let _range_info_from_plate: int option list -> (int * int * int) option = function
      | [] ->
         failwith "_range_info_from_plate:todo:1" (* wy: not obvious... *)
      | [Some size] ->
         Some (0, size, 1)
      | [Some size; Some subsize] when size = subsize ->
         Some (0, size, 1)
      | [Some size; _] ->
         failwith "_range_info_from_plate:todo:2" (* wy: not obvious... *)
      | _ ->
         failwith "_range_info_from_plate:error:1"

    (* Helper functions: tensor-size transformers.
     * Function list:
     * - tensor_gen{1,2}_get_type
     * - tensor_arange_get_type
     * - tensor_access_get_type
     * - tensor_cat_get_type
     * - tensor_{matmul, transpose, indexselect}_get_type
     * - tensor_{affinegrid, gridsample}_get_type *)

    (* compute tensor size from arg of tensor gen funcs:
     * torch.{tensor, FloatTensor, LongTensor} *)
    (* wy: for now, we don't use t in the function,
     *     but may consider using it later, e.g., for constant propagation. *)
    let tensor_gen1_get_type (t: t) (arg: expr): exp_ty =
      let rec size: expr -> tensor_size_ty = function
        | e when is_real_exp t e -> (* size(n) = [] *)
           Some [] 
        | List(es) -> (* size([e1; e2; ...; en]) = n :: size(e1) *)
           let tsty_inner = size (List.hd es) in
           let tsty_outer = Some [Some(List.length es)] in
           ITU.tensor_size_ty_concat tsty_outer tsty_inner
        | _ -> None in
      ET_tensor (size arg)

    (* compute tensor size from arg of tensor gen funcs:
     * torch.{rand, zeros, ones} *)
    let tensor_gen2_get_type (t: t) (arg: expr): exp_ty =
      let size: expr -> tensor_size_ty = function
        | List(es) -> Some (List.map IU.expr_to_int_opt es)
        | _        -> None in
      ET_tensor (size arg)

    let tensor_arange_get_type (t: t) (args: expr list): exp_ty =
      let args_iopt = List.map IU.expr_to_int_opt args in
      let ts = 
        match _range_info_from_range args_iopt with
        | None           -> Some [None]
        | Some (l, u, s) -> Some [Some ((u-l)/s)] in
      ET_tensor ts

    let rec _ts_access_index (t: t) (szs: int option list) (inds: expr list)
            : tensor_size_ty =
      let normalize_index (i: int) (len: int): int =
        (* post: 0 <= res <= len *)
        if      i < -len then 0
        else if i <    0 then i + len
        else if i <= len then i
        else                  len in
      let slice_size ((l,u,s): int * int * int) (len: int): int =
        let l = normalize_index l len in
        let u = normalize_index u len in
        let res = (u-l)/s in
        if res < 0 then 0 else res in
      match szs, inds with
      | [], [] -> Some []
      | [],  _ -> raise (Must_error "_ts_access_index: tensor access error")
      |  _, [] -> Some szs
      | sz :: szs_tl, ind :: inds_tl ->
         let new_ts =
           match ind with
           | List([el; eu; es]) ->
              begin
                match sz with
                | None -> Some [None]
                | Some(len) ->
                   match el, eu, es with
                   | Num(Int(l)), Num(Int(u)), Num(Int(s)) ->
                      Some [Some (slice_size (l, u,   s) len)]
                   | Num(Int(l)), Nil,         Num(Int(s)) ->
                      Some [Some (slice_size (l, len, s) len)]
                   | _ ->
                      Some [None]
              end
           | _ -> 
              match (get_exp_ty t ind) with
              | ET_bool | ET_num _ -> Some []
              | ET_tensor ts -> ts
              | ET_nil | ET_plate _ | ET_range _ | ET_distr _ | ET_fun _ ->
                 raise (Must_error "_ts_access_index: tensor access error")
              | ET_unknown -> None in
         let new_ts_tl = _ts_access_index t szs_tl inds_tl in
         ITU.tensor_size_ty_concat new_ts new_ts_tl
      
    let tensor_access_get_type (t: t) (e: expr) (idx_lst: expr list): exp_ty =
      match get_exp_ty t e with
      | ET_bool | ET_nil | ET_num _ | ET_plate _ | ET_range _ | ET_distr _ | ET_fun _ ->
          raise (Must_error "tensor_access_get_type: tensor access error")
      | ET_unknown | ET_tensor None -> 
          (* hy: using partial correctness, we may assume that
           * the result is a tensor. *)
          ET_tensor None 
      | ET_tensor (Some size_lst) ->
          ET_tensor (_ts_access_index t size_lst idx_lst)

    let _ts_add_along_dim (ts1: tensor_size_ty) (ts2: tensor_size_ty) (dim: int): tensor_size_ty =
      let rec aux (d: int) (s1: int option list) (s2: int option list) (ind: int): int option list =
        match s1, s2 with
        | [], [] -> []
        | hd1::tl1, hd2::tl2 ->
           let hd_new = 
             if ind <> d then
               (* make size precise *)
               match hd1, hd2 with
               | None, None -> None
               | None, Some(n2) -> Some(n2)
               | Some(n1), None -> Some(n1)
               | Some(n1), Some(n2) ->
                  if n1=n2 then Some(n1)
                  else failwith "_ts_add_along_dim: wrong tensor sizes 1"
             else
               (* add size *)
               match hd1, hd2 with
               | Some(n1), Some(n2) -> Some (n1+n2)
               | _ -> None
           in hd_new :: (aux d tl1 tl2 (ind+1))
        | _ -> failwith "_ts_add_along_dim: wrong tensor sizes 2" in
      let get_dim s = 
        if dim >= 0 then dim else (List.length s + dim) in
      match ts1, ts2 with
      | Some(s1), Some(s2) -> Some (aux (get_dim s1) s1 s2 0)
      | None, Some(s2) -> Some (aux (get_dim s2) (list_repeat (List.length s2) None) s2 0)
      | Some(s1), None -> Some (aux (get_dim s1) (list_repeat (List.length s1) None) s1 0)
      | None, None -> None
      
    let tensor_cat_get_type (t: t) (args: expr list) (dim: int): exp_ty =
      let ety_to_tsty: exp_ty -> tensor_size_ty = function
        (* get tensor size from (ety:exp_ty), assuming that e is a tensor. *)
        | ET_tensor(ts) -> ts
        | ET_unknown -> None
        | ET_bool | ET_nil | ET_num _ | ET_plate _ | ET_range _ | ET_distr _ | ET_fun _ ->
           raise (Must_error "tensor_cat_get_type: non-tensor args to tensor.cat") in 
      let exp_tys = List.map (get_exp_ty t) args in
      let tens_tys = List.map ety_to_tsty exp_tys in
      let new_ts = 
        List.fold_left 
          (fun acc0 ety0 -> _ts_add_along_dim acc0 ety0 dim)
          (List.hd tens_tys) 
          (List.tl tens_tys) in
      ET_tensor new_ts 

    let tensor_matmul_get_type (t: t) (tens1: expr) (tens2: expr): exp_ty =
      let ts = 
        match get_exp_ty t tens1, get_exp_ty t tens2 with
        | ET_tensor(Some[_]), ET_tensor(Some[_]) -> Some[]
        | ET_tensor(Some[i;_]), ET_tensor(Some[_;k]) -> Some[i;k]
        | _ -> None (* wy: more cases need to be implemented. *) in
      ET_tensor ts

    let tensor_transpose_get_type (t: t) (tens: expr) (dim0: int) (dim1: int): exp_ty =
      let ts = 
        match get_exp_ty t tens with
        | ET_tensor(Some(size_l)) ->
           let n0, n1 = List.nth size_l dim0, List.nth size_l dim1 in
           Some (list_replace dim0 n1 size_l |> list_replace dim1 n0)
        | _ -> None in
      ET_tensor ts

    let tensor_indexselect_get_type (t: t) (input: expr) (dim: int) (index: expr): exp_ty =
      let ts =
        match get_exp_ty t input with
        | ET_tensor(Some input_sz) ->
           begin
             match get_exp_ty t index with
             | ET_tensor(Some [ind_sz]) ->
                Some (List.mapi (fun i x -> if i=dim then ind_sz else x) input_sz)
             | _ ->
                None
           end
        | _ -> None in
      ET_tensor ts

    let tensor_affinegrid_get_type (t: t) (theta: expr) (size: expr list): exp_ty =
      let (n, h, w) = (IU.expr_to_int_opt (List.nth size 0),
                       IU.expr_to_int_opt (List.nth size 2),
                       IU.expr_to_int_opt (List.nth size 3)) in
      ET_tensor (Some [n; h; w; Some 2])

    let tensor_gridsample_get_type (t: t) (input: expr) (grid: expr): exp_ty =
      let (n, c, h_out, w_out) = 
        match get_exp_ty t input, get_exp_ty t grid with
        | ET_tensor(Some [n;c;_;_]), ET_tensor(Some [n2;h_out;w_out;_]) ->
           (n, c, h_out, w_out)
        | _ ->
           (None, None, None, None) in
      ET_tensor (Some [n; c; h_out; w_out])

    (* Helper functions:
     * Function list:
     * - infer_fun_ty
     * - refine_expr_opt 
     * - refine_dist_kind 
     * - can_nil *)
    let infer_fun_ty fstr args =
      let tens_resize_get_params = function
        | ("nn.Linear", [Num (Int i_dim); Num (Int o_dim)]) -> 
           Some ([Some i_dim], [Some o_dim], [])
        | ("nn.LSTMCell", [Num (Int i_dim); Num (Int h_dim)]) ->
           (* wy: the part [Some 2] is in fact incorrect in that 
            * nn.LSTMCell(i,h)(input,(h0,c0)) returns a tuple (h1,c1),
            * not a tensor [h1,c1]. But to make our analyzer handle it,
            * we assume for now that the return value is a tensor [h1,c1]. *)
           Some ([Some i_dim], [Some h_dim], [Some 2])
        | ("nn.RNN", [Num (Int input_size); Num (Int hidden_size); _;
                      _; _; _; bidirectional]) ->
           (* wy: The below code is actually unsound in that
            * `nn.RNN(...)(input,h0)' return a tuple `(output,hn)',
            * where the tensor size of `output' and `hn' are *different*.
            * The below assumes that `hn' has the tensor size of output.
            * This works for DMM testcase because `hn' is not used in the testcase at all.
            * In the future, it needs to be corrected by modifying the semantics of 
            * the type `FT_tens_resize'. *)
           (* NOTE: The order of nn.RNN's argument is set as follows:
            *   nn.RNN(input_size: int, hidden_size: int, num_layers: int,
            *          bias: bool, batch_first: bool, dropout: float, bidirectional: bool)
            *          with `non_linearity: string' in kwargs. *)
           let num_directions = if bidirectional = True then 2 else 1 in
           Some ([Some input_size], [Some (hidden_size * num_directions)], [Some 2])
        | _ -> 
            None
      in
      if List.mem fstr tens_identity_cls then 
        tens_id_fn_ty
      else if List.mem fstr tens_resize_cls then
        match tens_resize_get_params (fstr, args) with
        | None -> FT_top
        | Some (l1, l2, l3) -> FT_tens_resize(l1, l2, l3)
      else
        FT_top

    let refine_expr_opt (t: t) (e_opt: expr option): expr option =
      match e_opt with
      | None | Some Nil -> 
          e_opt
      | Some e -> 
          let e' = eval_exp t e in
          if e = e' then e_opt else Some e'

    let refine_dist_kind (args: expr list) (args_ty_l: exp_ty list)
           (dk: dist_kind): dist_kind =
      let get_last_from_ts (): int option = 
        try
           let ts = ITU.exp_ty_do_broadcast args_ty_l in
           match ts with
           | None | Some [] -> None 
           | Some l -> List.hd (List.rev l)
        with
        | ITU.Broadcast_failure -> None
      in
      match dk with
      | Normal | Exponential | Gamma | Beta | Poisson | Bernoulli | Delta -> dk
      (* Dirichlet, Categorical, OneHotCategorical *)
      | Dirichlet None     -> Dirichlet (get_last_from_ts())
      | Dirichlet (Some _) -> dk
      | Categorical None     -> Categorical (get_last_from_ts())
      | Categorical (Some _) -> dk
      | OneHotCategorical None     -> OneHotCategorical (get_last_from_ts())
      | OneHotCategorical (Some _) -> dk
      (* Uniform *)
      | Uniform None ->
          (match args with
           | [Num (Float l); Num (Float u)] -> Uniform (Some (l, u))
           | _ -> dk)
      | Uniform (Some _) -> dk
      (* Subsample *)
      | Subsample (None, _)
      | Subsample (_, None) ->
          let args_simp = list_drop_trailing_elt Nil args in
          let (size, subsize) = 
            match args_simp with
            | [] ->
               Some (-1), Some (-1)
               (* FACT: if p is a plate with no `size', then `p.size = p.subsample_size = -1'. *)
            | [Num (Int size)] ->
               Some size, Some size
            | [Num (Int size); Num (Int subsize)] ->
               Some size, Some subsize
            | _ ->
               None, None in
          Subsample (size, subsize)
      | Subsample (Some _, Some _) -> dk

    (* returns true if the argument can be nil, and false if it cannot. *)
    let can_nil (t: t) (obs: expr) =
      match (get_exp_ty t obs) with
      | ET_bool -> false
      | ET_nil -> true
      | ET_num _ | ET_plate _ | ET_range _ | ET_distr _ | ET_fun _ | ET_tensor _ -> false
      | ET_unknown -> true

    (* Main function: post-conditions *)
    let rec eval ac (t: t): t =
      match ac with
      | Assume e ->
          let ac_new = if !AM.sim_assm then Assume (eval_exp t e) else ac in
          { t with
            t_u_num = D_num.eval ac_new t.t_u_num }
      | Assert e ->
          let ac_new = if !AM.sim_assm then Assume (* wy: is this correct? why not Assert? *)
                                              (eval_exp t e) else ac in
          { t with
            t_u_num = D_num.eval ac_new t.t_u_num }

      | Sample (x, name, dist, args, Some obs) when not (can_nil t obs) ->
          eval (Assn(x,obs)) t
      | Sample (x, name, dist, args, obs_opt) ->
          (* compute: dist_size_opt *)
          let (dist_kind, dist_trans_l) = dist in
          let arg_ty_list = List.map (fun e -> get_exp_ty t e) args in
          let new_dist_kind = refine_dist_kind args arg_ty_list     dist_kind in
          let dist_size_opt = ITU.DistSize.get_ty   arg_ty_list new_dist_kind
                              |> ITU.DistSize.apply_trans_l dist_trans_l
                              |> ITU.DistSize.apply_broadcast t.t_bcast in
          (* update dim for x: t_vtyp, t_u_num (for TENS_SZ) *)
          let t =
            match dist_size_opt with
            | None -> var_becomes_other_havoc x t
            | Some (batch_size, event_size) ->
                let ts = Some (batch_size @ event_size) in
                var_becomes_tens_size x ts (true, None) t in
          (* update dim for name: t_samp{var,idx}, t_u_typ, t_u_num (for LOW/HIGH, or zone) *)
          let t = update_name_sets t name in
          (* compute: new_ac *)
          let new_dist = (new_dist_kind, dist_trans_l) in
          let new_obs_opt = refine_expr_opt t obs_opt in
          let new_ac = Sample(x, name, new_dist, args, new_obs_opt) in
          (* update dim for name: t_u_typ *)
          let t = update_distty t new_ac in
          t

      | Assn (x, e) ->
          begin
            match get_exp_ty t e with
            | ET_tensor ts ->
                begin
                  match e with
                  | Name y when y <> x -> var_becomes_tens_var x y ts t
                  | _                  -> var_becomes_tens_size x ts (true, None) t
                end
            | ET_num nt -> var_assign_num x nt (Some ac) t
            | ET_plate pt ->
               let t = var_becomes_plate x pt t in
               (* copy all fields of a plate `pt' (i.e., pt.{_indices, dim})
                  into that of `x' *)
               let acmds =
                 match e with
                 | Name e_name->
                    [Assn(NM.plate_indices_name x, Name(NM.plate_indices_name e_name));
                     (*Assn(NM.plate_dim_name     x, Name(NM.plate_dim_name     e_name))*)]
                 | _ ->
                    [AssnCall(NM.plate_indices_name x, Name("RYLY_tens"), [], []);
                     (*AssnCall(NM.plate_dim_name     x, Name("RYLY_int"), [], [])*)] in
               List.fold_left (fun acc0 acmd0 -> eval acmd0 acc0) t acmds
            | ET_range rt -> var_becomes_range x rt t
            | ET_distr dt -> var_becomes_distr x dt t
            | ET_fun ft -> var_becomes_fun x ft t
            | ET_nil -> var_becomes_nil x t
            | ET_bool | ET_unknown -> var_becomes_other_havoc x t
          end

      | AssnCall (x, fexpr, args, kwargs) ->
          let args = List.map (eval_exp t) args in

          (* tens fns of known type *)
          match fexpr with
          | _ when is_fun_exp t fexpr ->
              begin
                match args with
                | arg :: _ ->
                    (* wy: temporary fix. changed from `[arg] ->' to `arg :: _'
                     * to handle both nn.Linear and nn.{LSTMCell, RNN}. *)
                    let fexpr_ty = get_exp_ty t fexpr in
                    let arg_ty = get_exp_ty t arg in
                    (match fexpr_ty, arg_ty with
                    | ET_fun ft, ET_tensor ts ->
                        let new_ts = ITU.fun_ty_apply ft ts in
                        var_becomes_tens_size x new_ts (true, None) t
                    | _ ->
                        var_becomes_other_havoc x t)
                | _ ->
                    var_becomes_other_havoc x t
              end

          (* tens creation fns 1 *)
          | Name fstr when List.mem fstr
                             ["torch.tensor"; "torch.FloatTensor"; "torch.LongTensor"] ->
              (* https://pytorch.org/docs/stable/torch.html#torch.tensor
               * https://pytorch.org/docs/stable/tensors.html#torch-tensor *)
              let arg_normed: expr =
                match args with
                | [arg] -> arg
                | _ -> failwith "torch.{gen1}: with illegal arguments" in
              begin
                match tensor_gen1_get_type t arg_normed with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> failwith "torch.{gen1}: unreachable"
              end
          (* tens creation fns 2 *)
          | Name fstr when List.mem fstr
                             ["torch.rand"; "torch.randn";
                              "torch.zeros"; "torch.ones"] ->
              (* https://pytorch.org/docs/stable/torch.html#torch.rand
               * https://pytorch.org/docs/stable/torch.html#torch.randn
               * https://pytorch.org/docs/stable/torch.html#torch.zeros
               * https://pytorch.org/docs/stable/torch.html#torch.ones *)
              let arg_normed: expr =
                match args with
                | [] -> failwith "torch.{gen2}: with illegal arguments"
                | [arg] when (is_int_exp t arg) -> List [arg]
                | [arg] -> arg
                | _ -> List(args) in
              begin
                match tensor_gen2_get_type t arg_normed with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> failwith "torch.{gen2}: unreachable"
              end
          (* tens reshaping fns *)
          | Name fstr when List.mem fstr
                             ["torch.reshape"; "torch.Tensor.expand"; "torch.Tensor.view"] ->
              (* https://pytorch.org/docs/stable/torch.html#torch.reshape
               * https://pytorch.org/docs/stable/tensors.html#torch.Tensor.expand
               * https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view *)
             let arg_normed: expr =
                match args with
                | [] | [_] -> raise (Must_error "eval: torch.reshape with < 2 arguments")
                | _ :: [arg] when (is_int_exp t arg) -> List [arg]
                | _ :: [arg] -> arg
                | _ :: args_tl  -> List(args_tl) in
              begin
                match tensor_gen2_get_type t arg_normed with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> failwith "torch.{reshape}: unreachable"
              end
          (* tens arange *)
          | Name "torch.arange" ->
             (* https://pytorch.org/docs/stable/torch.html#torch.arange *)
             begin
               match tensor_arange_get_type t args with
               | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
               | _ -> failwith "torch.arange: unreachable"
             end
          (* tens index_select *)
          | Name "torch.index_select" ->
             (* https://pytorch.org/docs/stable/torch.html#torch.index_select *)
             let (input, dim, index): expr * int * expr =
                match args with
                | [input; Num(Int(dim)); index] -> (input, dim, index)
                | _ -> failwith "torch.cat with unexpected arguments" in
             begin
               match tensor_indexselect_get_type t input dim index with
               | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
               | _ -> failwith "torch.cat: unreachable"
             end
          (* tens indexing *)
          | Name "access_with_index" ->
              let (e_arr, indices): expr * expr list =
                match args with
                | [] | _::[] -> failwith "access_with_index with < 2 arguments"
                | e_arr :: [List(indices)] -> e_arr, indices
                | e_arr :: indices -> e_arr, indices in
              begin
                match tensor_access_get_type t e_arr indices with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> var_becomes_other_havoc x t
              end
          (* tens concat *)
          | Name "torch.cat" ->
             (* https://pytorch.org/docs/stable/torch.html#torch.cat *)
             let (tensors, n): expr list * int =
                match args with
                | [List(tensors); Num(Int(n))] -> (tensors, n)
                | _ -> failwith "torch.cat with unexpected arguments" in
             begin
               match tensor_cat_get_type t tensors n with
               | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
               | _ -> failwith "torch.cat: unreachable"
             end
          (* tens matmul *)
          | Name "torch.matmul" ->
             (* https://pytorch.org/docs/stable/torch.html#torch.matmul *)
             let (e1, e2): expr * expr =
                match args with
                | [e1; e2] -> e1, e2
                | _ -> failwith "torch.matmul with unexpected arguments" in
             begin
               match tensor_matmul_get_type t e1 e2 with
               | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
               | _ -> failwith "torch.matmul: unreachable"
             end
          (* tens transpose *)
          | Name fstr when List.mem fstr
                             ["torch.transpose"; "torch.Tensor.transpose"] ->
             (* https://pytorch.org/docs/stable/torch.html#torch.transpose *)
             let (e, dim0, dim1): expr * int * int =
                match args with
                | [e; Num(Int(dim0)); Num(Int(dim1))] -> e, dim0, dim1
                | _ -> failwith "torch.transpose with unexpected arguments" in
             begin
               match tensor_transpose_get_type t e dim0 dim1 with
               | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
               | _ -> failwith "torch.transpose: unreachable"
             end

          (* nn constructors *)
          | Name fstr when List.mem fstr
                             (tens_identity_cls @ tens_resize_cls) ->
              let ft = infer_fun_ty fstr args in
              var_becomes_fun x ft t
          (* nn fns *)
          | Name "F.affine_grid" ->
             (* https://pytorch.org/docs/stable/nn.html#torch.nn.functional.affine_grid *)
              let (theta, size): expr * expr list =
                match args with
                | [theta; List(size)] when List.length size = 4 -> theta, size
                | _ -> failwith "F.affine_grid with unexpected arguments" in
              begin
                match tensor_affinegrid_get_type t theta size with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> failwith "F.affine_grid: unreachable"
              end
          | Name "F.grid_sample" ->
             (* https://pytorch.org/docs/stable/nn.html#torch.nn.functional.grid_sample *)
              let (input, grid): expr * expr =
                match args with
                | [input; grid] -> input, grid
                | _ -> failwith "F.grid_sample with unexpected arguments" in
              begin
                match tensor_gridsample_get_type t input grid with
                | ET_tensor ts -> var_becomes_tens_size x ts (true, None) t
                | _ -> failwith "F.grid_sample: unreachable"
              end

          (* "D": pyro distr constructors *)
          | Name fstr when List.mem fstr IC.dist_kind_list ->
             let dk = IC.to_dist_kind fstr in
             let arg_ty_list = List.map (fun e -> get_exp_ty t e) args in
             let arg_tsty_list = List.map (function | (ET_tensor ts) -> ts
                                                    | _ -> None) arg_ty_list in
             let dt = Some (dk, arg_tsty_list) in
             var_becomes_distr x dt t
          (* "D.log_prob" *)
          | Name fstr when
                 (* when fstr = dk_str ^ ".log_prob", where dk_str \in IC.dist_kind_list *)
                 (match String.split_on_char '.' fstr with
                  | [dk_str; "log_prob"] -> List.mem dk_str IC.dist_kind_list
                  | _ -> false) ->
             let (d_obj, logprob_arg): expr * expr =
               match args with
               | [e1; e2] -> e1, e2
               | _ -> failwith "D.log_prob with unexpected arguments" in
             let ts =
               match get_exp_ty t d_obj with
               | ET_distr (Some (d_kind, d_arg_ts_list)) ->
                  (* wy: for now, assume that `dk_str' and `d_kind' denote the same dist_kind. *)
                  (* dist_size_opt *)
                  let d_arg_ty_list = List.map (fun ts -> ET_tensor ts) d_arg_ts_list in
                  let new_d_kind = refine_dist_kind []    d_arg_ty_list     d_kind in
                  let dist_size_opt = ITU.DistSize.get_ty d_arg_ty_list new_d_kind in
                  (* logprob_arg_ty *)
                  let logprob_arg_ty = get_exp_ty t logprob_arg in
                  (* result *)
                  ITU.DistSize.get_logprob_ts logprob_arg_ty dist_size_opt
               | _ -> None in
             var_becomes_tens_size x ts (true, None) t 

          (* pyro fns *)
          | Name "pyro.param" ->
              (* similar to transfer func of Assn (x, e), but this considers only tensor inputs *)
              let e: expr =
                match args with
                | [_; e] -> e
                | _ -> failwith "pyro.param with unexpected arguments" in
              begin
                match get_exp_ty t e with
                | ET_tensor ts ->
                   begin
                     match e with
                     | Name y when y <> x -> var_becomes_tens_var x y ts t
                     | _                  -> var_becomes_tens_size x ts (true, None) t
                   end
                | _ -> var_becomes_other_havoc x t
              end
          | Name "pyro.plate" ->
              let args_simp   = List.map (eval_exp t) args in
              let kwargs_simp = List.map (fun (i0,e0) -> (i0, eval_exp t e0)) kwargs in
              let pt_s_args = List.tl args_simp |> list_drop_trailing_elt Nil
                              |> List.map IU.expr_to_int_opt (* |> Some *) in
              let dim_n: int option =
                try 
                  let (_, dim_e): keyword =
                    List.find (fun (name,e) -> name = Some "dim") kwargs_simp in
                  match dim_e with
                  | Num (Int s) -> Some s
                  | _           -> None
                with Not_found -> Some 0 in
              (* FACT: if p is a plate with no `dim', then `p.dim = None' initially. *)
              (* Here `Some 0' denotes `None' in python,
               * to handle the case when a plate `p' is created without `dim' info.
               * wy: It would be ok to use `Some 0' to represent `None' in this case,
               *     since `with p: ...' gives an error if `p.dim >= 0'. *)
              let pt_n_args = SM.add "dim" dim_n SM.empty in
                (* let f m (i,e) =
                   match i with
                   | None -> m
                   | Some i0 when i0 = "dim" -> 
                       (match e with
                       | Num (Int s) -> SM.add i0 (Some s) m
                       | _           -> SM.add i0  None    m)
                   | _ -> m in
                List.fold_left f SM.empty kwargs_simp in *)
              let pt: plate_ty = Some (pt_s_args, pt_n_args) in
              (* let pt: plate_ty =
                match pt_s_args with
                | None -> None
                | Some pt_args_val -> Some (pt_args_val, pt_kwargs) in *)
              var_becomes_plate x pt t 

          (* misc *)
          | Name "len" ->
              (* - drops all information about x
               * - makes x int (with no information about its value)
               * - asserts that x is non-negative *)
              begin
                let t = var_assign_num x NT_int None t in
                match args with
                | [Name y as e] when x <> y && is_tensor_exp t e ->
                    var_assign_num x NT_int
                      (Some (Assn (x, Name (dim_tens_size_make y)))) t
                | _ ->
                    var_assign_num x NT_int
                      (Some (Assume (Comp (GtE, Name x, Num (Int 0))))) t
             end
          | Name "float" ->
              var_assign_num x NT_real None t
          | Name "RYLY_int" ->
              var_assign_num x NT_int  None t
          | Name "RYLY_real" ->
              var_assign_num x NT_real None t
          | Name "RYLY_tens" ->
              var_becomes_tens_havoc x t
          | Name "update_with_index"
          | Name "update_with_field" ->
             begin
               match args with
               | Name(y) :: _ when x=y -> t
               | _ -> var_becomes_other_havoc x t
             end
          | Name "range" ->
              let args_simp = List.map (eval_exp t) args in              
              let rt: range_ty = Some (List.map IU.expr_to_int_opt args_simp) in
              var_becomes_range x rt t
          | _ ->
              var_becomes_other_havoc x t
              (* - drops out all information about x
               * - but type of x remains unknown so we may have crashes later *)

    let enter_with ((e_ctx, e_opt): withitem) (t: t): t =
      match (get_exp_ty t e_ctx) with
      | ET_plate (Some (pt_s_args, pt_n_args)) ->
         begin
           let e_ctx_name =
             match e_ctx with
             | Name str -> str
             | _ -> failwith "enter_with: error 1" in
           (* 1. eval `e_opt_name = __@@_indices(e_ctx_name)'. *)
           let acmd =
             match e_opt with
             | None -> Assume(True)
             | Some (Name e_opt_name) ->
                Assn(e_opt_name, Name(NM.plate_indices_name e_ctx_name))
             | Some _ -> failwith "enter_with: error 2" in
           let t = eval acmd t in
           (* 2. compute `dim' of e_ctx_name, and
            *    update `e_ctx''s `plate_ty' with the new `dim'. *)
           (* FACT: once `dim' of a plate is determined, it doesn't change anymore
            *       even if it is used in another `with' clause. *)
           let dim_cur_pre : int option  =
             try SM.find "dim" pt_n_args
             with Not_found -> failwith "enter_with: dim not found" in
           let dim_cur =
             if dim_cur_pre = Some 0 then
               (* The case when `e_ctx_name.dim = None'.
                * So we need to find an appropriate value for it. *)
               Some (ITU.next_dim_from_broadcast_info t.t_bcast)
             else
               dim_cur_pre in
           let pt_new: plate_ty = Some (pt_s_args, SM.add "dim" dim_cur pt_n_args) in
           let t = var_becomes_plate e_ctx_name pt_new t in
           (* 3. append new bcast_info (generated by e_ctx) to t.t_bcast. *)
           let bcast_size_opt : int option =
             match pt_s_args with
             | [] -> None
             | [Some size] -> Some size
             | [Some size; Some subsize] -> Some subsize
             | _ -> failwith "enter_with: error 3" in
           let bcast_new : (int * int option) list =
             match bcast_size_opt with
             | None -> [] (* FACT: a plate with no `size' doesn't do any automatic broadcasting. *)
             | Some bcast_size -> [(bcast_size, dim_cur)] in
           let t =
             { t with
               t_bcast = t.t_bcast @ bcast_new } in
           t
         end
      | ET_plate None -> failwith "enter_with: error 4"
      | _ -> t
      
    let exit_with ((e_ctx, _): withitem) (t: t): t =
      match (get_exp_ty t e_ctx) with
      | ET_plate (Some (pt_s_args, _)) ->
         begin
           (* 2. remove added bcast_info (generated by e_ctx) from t.t_bcast. *)
           let rm_bcast_num = 
             match pt_s_args with
             | [] -> 0
             | [Some _] | [Some _; Some _] -> 1
             | _ -> failwith "exit_with: error 3" in
           let bcast_new = list_drop_last rm_bcast_num t.t_bcast in
           let t =
             { t with
               t_bcast = bcast_new } in
           t
         end
      | ET_plate None -> failwith "exit_with: error 4"
      | _ -> t

    (*******)
    (* sat *)
    (*******)
    (* tries to prove that a condition holds
     * soundness: if sat e t returns true, all states in gamma(t) satisfy e *)
    let sat (e: expr) (t: t): bool =
      match e with
      | UOp(SampledStr, Name(x)) ->
          SS.mem x t.t_sampvar.m_must
      | UOp(SampledStrFmt, Name(x)) ->
          SS.mem x t.t_sampidx.m_must
      | e -> D_num.sat e t.t_u_num

    (* helper functions for is_related *)
    let has_exact_rv_sets (t: t): bool =
      are_same_sets t.t_sampidx.m_may t.t_sampidx.m_must
      && are_same_sets t.t_sampvar.m_may t.t_sampvar.m_must

    let is_const_intvl ((inf, sup): int * int): int option =
      if inf = sup then Some inf else None

    (* wy: this func will be removed if we decide to use zone only. *)
    let has_same_low_high_one (t: t): bool =
      let has_same_const_val (dim_make: maymust -> string -> string) (s: string): bool =
        let iopt_must = is_const_intvl (D_num.bound_var_apron (dim_make Must s) t.t_u_num) in
        let iopt_may  = is_const_intvl (D_num.bound_var_apron (dim_make May  s) t.t_u_num) in
        opt_some_eq iopt_must iopt_may in
      let name_must = get_name_with_1d_idx t.t_sampidx.m_must in
      SS.for_all (has_same_const_val dim_sample_low_make) name_must
      && SS.for_all (has_same_const_val dim_sample_high_make) name_must

    (* wy: this func will be removed if we decide to use zone only. *)
    let has_same_low_high_two (t1: t) (t2: t): bool =
      let has_same_const_val (dim_make: maymust -> string -> string) (s: string): bool =
        let iopt_1 = is_const_intvl (D_num.bound_var_apron (dim_make May s) t1.t_u_num) in
        let iopt_2 = is_const_intvl (D_num.bound_var_apron (dim_make May s) t2.t_u_num) in
        opt_some_eq iopt_1 iopt_2 in
      let name_may = get_name_with_1d_idx t1.t_sampidx.m_may in
      SS.for_all (has_same_const_val dim_sample_low_make) name_may
      && SS.for_all (has_same_const_val dim_sample_high_make) name_may

    (* checks whether t1 and t2 describe the sets of random variable names
     * precisely, and those sets are identical and associated with same
     * distribution types, except that t1 may have random variables from
     * subsampling.
     *
     * - t1 is the analysis result for a model.
     * - t2 is the analysis result for a guide.*)
    let is_related (t1: t) (t2: t): bool =
      (* wy: The condition 2 is checked by D_typ.is_related as well,
       *     assuming that rv's from Subsample _ are not indexed,
       *     i.e., the name of pyro.plate is not indexed. E.g.:
       *        for i in range(10):
       *          p[i] = pyro.plate("x_{}".format(i), ...)
       *     I leave it not commented out just in case. *)
      let name_must_1d = get_name_with_1d_idx t1.t_sampidx.m_must in
      let name_must_2d = get_name_with_2d_idx t1.t_sampidx.m_must in

      ignore name_must_1d;

      (* 1,2: t_sampidx, t_sampvar *)
      (* 1: (t1.t_sampidx.m_may = t1.t_sampidx.m_must /\
       *     t1.t_sampvar.m_may = t1.t_sampvar.m_must) /\
       *    (same for t2) *)
      let b1 = 
        has_exact_rv_sets t1
        && has_exact_rv_sets t2 in
      (* 2: t2.t_sampidx.m_must =         t1.t_sampidx.m_must /\
       *    t2.t_sampvar.m_must \subseteq t1.t_sampvar.m_must *)
      let b2 = 
        are_same_sets t2.t_sampidx.m_may t1.t_sampidx.m_may
        && SS.subset     t2.t_sampvar.m_may t1.t_sampvar.m_may in
      (* 3: t_u_typ *)
      let b3 = D_typ.is_related t1.t_u_typ t2.t_u_typ in
      (* 4: t_u_num (for LOW/HIGH) *)
      let b4 = 
        has_same_low_high_one t1
        && has_same_low_high_one t2
        && has_same_low_high_two t1 t2 in
      (* wy: for debug.
      let _ = 
        Printf.printf "b1-4 = %B %B %B %B\n" b1 b2 b3 b4;
        Printf.printf "name_must_1d: %a\n" (buf_to_channel buf_ss) name_must_1d;
        Printf.printf "name_must_2d: %a\n" (buf_to_channel buf_ss) name_must_2d in *)
      (* 5: t_u_num (for zones) *)
      let b5 =
        SS.for_all
          (fun s -> D_num.zone_is_related s t1.t_u_num t2.t_u_num)
          (* wy: later it should become `t1.t_sampidx.m_must' *)
          name_must_2d in
      (* wy: for debug.
         Printf.printf "b5 = %B\n" b5; *)
      b1 && b2 && b3 && b4 && b5        

    let range_info e t =
      match get_exp_ty t e with
      | ET_plate (Some (pt_s_args, _)) ->
         _range_info_from_plate pt_s_args
      | ET_range (Some rt_args) ->
         _range_info_from_range rt_args
      | _ -> None
      
  end: ABST_DOMAIN_NB)
