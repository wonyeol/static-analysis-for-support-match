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
 ** main.ml: launching of the analysis (with options already parsed) *)
open Lib

(* Temporary opens, to force compilation *)
open Pyobj_util
(* open Pyast_sig *)
open Pyast_util
open Pyast_dump
open Pyast_cast
(* open Ir_sig *)
open Ir_cast
open Ir_util
open Adom_sig
(* open Adom_kind *)
(* open Adom_state *)
open Adom_apron
open Adom_bot
open Adom_fib
open Adom_distty
open Ai_sig
open Analysis_sig

module IR = Ir_sig

let output = ref true

(** Expected outcome of a relational test case wrt. a default test *)
type test_oracle_r =
  | TOR_succeed (* the default test should succeed *)
  | TOR_fail    (* the default test should fail *)
  | TOR_error   (* must error *)

let string_of_test_oracle_r = function
  | TOR_succeed -> "TOR_succeed"
  | TOR_fail    -> "TOR_fail"
  | TOR_error   -> "TOR_error"


(** Default options *)
let aopts_default =
  { ao_do_num   = None;
    ao_input    = AI_pystring "x = 42";
    ao_debug_it = true;
    ao_wid_thr  = false;
    ao_zone     = false;
    ao_sim_assm = false }

let parse_code (i: int option) (input: analysis_input): IR.prog =
  (* get python_code *)
  let python_code : string =
      match input with
      | AI_pyfile fname -> read_file fname
      | AI_pystring str -> str in
  (* construction of the Py.Object.t using Pyml *)
  let pyobj = Pyobj_util.get_ast python_code in
  (* construction of the Pyast AST *)
  let pyast = Pyast_cast.pyobj_to_modl pyobj in
  (* construction of the IR AST *)
  let irast = Ir_cast.modl_to_prog pyast in
  if !output then
    begin
      let s =
        match i with
        | None -> ""
        | Some i -> Printf.sprintf "(%d)" i in
      Printf.printf "[IR%s]\n%a\n" s Ir_util.pp_prog irast
    end;
  irast

let run_analysis
    (debug: bool)
    (zone: bool)
    (dn: ad_num)
    (prog_aopts_list: (IR.prog * analysis_opts) list)
    (postconds: IR.expr list)
    (violations: IR.expr list) =
  let mod_mgr =
    match dn with
    | AD_box -> (module Adom_apron.PA_box:  APRON_MGR)
    | AD_oct -> (module Adom_apron.PA_oct:  APRON_MGR)
    | AD_pol -> (module Adom_apron.PA_pol:  APRON_MGR) in
  let module Mgr = (val mod_mgr: APRON_MGR) in
  let module Num   = (Adom_apron.Make( Mgr ): ABST_DOMAIN_NB_D) in
  let module Dist  = (Adom_distty.DistType: ABST_DOMAIN_NB_D) in
  let mod_zone =
    if zone then (module Adom_zone.MakeZone( Num ): ABST_DOMAIN_ZONE_NB)
    else (module Adom_zone.MakeId( Num ): ABST_DOMAIN_ZONE_NB) in
  let module NumZ  = (val mod_zone: ABST_DOMAIN_ZONE_NB) in
  let module NumF  = (Adom_fib.Make( NumZ )( Dist ): ABST_DOMAIN_NB) in
  let module NumFB = (Adom_bot.Make( NumF ): ABST_DOMAIN_B) in
  let module AiNum = (Ai_make.MakeAnalysis( NumFB ): ANALYSIS) in
  let pass = ref true in
  let prev_x = ref None in
  let check_post x e =
    if not (AiNum.Ad.sat e x) then
      begin
        Printf.printf "Post-cond %a not satisfied\n" Ir_util.pp_expr e;
        pass := false
      end in
  let check_violation x e =
    if AiNum.Ad.sat e x then
      begin
        Printf.printf "Violation %a not satisfied\n" Ir_util.pp_expr e;
        pass := false
      end in
  let check_relatedness x =
    match !prev_x with
    | Some x0 when not (AiNum.Ad.is_related x0 x) ->
        Printf.printf "[CHECK] Not related\n";
        pass := false
    | _ -> () in
  List.iteri
    (fun i (prog,aopts) ->
      if !output then
        Printf.printf "[Analysis init(%d): %s]\n\n" i AiNum.analysis_name;
      Ai_make.debug := debug && aopts.ao_debug_it;
      Ai_make.wid_thr := aopts.ao_wid_thr;
      Ai_make.sim_assm := aopts.ao_sim_assm;
      let x, _, _ = AiNum.analyze prog in
      if !output then
        Printf.printf "[Analysis output(%d)]\n%a\n" i AiNum.pp x;
      List.iter (check_post x) postconds;
      List.iter (check_violation x) violations;
      check_relatedness x;
      prev_x := Some x
    ) prog_aopts_list;
  !pass

(** Master function used for benchmarks for relational and non-relational
 ** properties. *)
(* xr's comment:
 * - by default, use the first string in the test array;
 * - if an input file is given, use that instead;
 *
 * hy's comment:
 * - when aopts_list has more than two elements, the type of analysis
 *   is determined by the ao_do_num field of the first aopts. Quick but
 *   ugly trick.
 *
 * For each program, the test goes through the following steps:
 * 1. parse with pyml
 * 2. convert into ir
 * 3. run the analysis that was selected
 * 4. check whether the analysis result implies the given postconditions
 * 5. check whether the analysis result does not imply any of the given
 *    violations
 * 6. If the previous analysis result exists, check whether the previous
 *    and current results are related by the is_related routine.
 *
 * Then, it returns true if and only if all these checks succeed for all
 * given programs. *)
let start
    (debug: bool) (* global debug switch --false: no debug, nowhere *)
    (aopts_list: analysis_opts list)
    (postconds: IR.expr list)
    (violations: IR.expr list) =
  match aopts_list with
  | [] -> true
  | aopts::_ ->
      let zone = aopts.ao_zone in
      begin
        match aopts.ao_do_num with
        | None ->
            Printf.printf "\n";
            true
        | Some dn ->
            let irast_list =
              List.mapi (fun i aopts -> parse_code (Some i) aopts.ao_input)
                aopts_list in
            let irast_aopts_list = List.combine irast_list aopts_list in
            run_analysis debug zone dn irast_aopts_list postconds violations
      end

(** Starts a benchmark for checking non-relational properties found
 ** by the analysis *)
let start_nr
    (debug: bool) (* global debug switch --false: no debug, nowhere *)
    (aopts: analysis_opts)
    (postconds: IR.expr list)
    (violations: IR.expr list): bool =
  start debug [aopts] postconds violations

(** Starts a benchmark for checking relational properties found
 ** by the analysis *)
let start_r
    (debug: bool) (* global debug switch --false: no debug, nowhere *)
    (aopts1: analysis_opts)
    (aopts2: analysis_opts)
    : test_oracle_r -> bool * string =
  function
  | TOR_succeed ->
      begin
        try
          let b = start debug [aopts1;aopts2] [] [] in
          b, (if b then "[Ok(ok!)]" else "[Ok(ko!)]")
        with IR.Must_error msg ->
          Printf.printf "[MUST ERROR: UNEXPECTED] %s\n" msg;
          false, ""
      end
  | TOR_fail    ->
      begin
        try
          let b = not (start debug [aopts1;aopts2] [] []) in
          b, (if b then "[Ok(ko!)]" else "[Ko(ko!)]")
        with IR.Must_error msg ->
          Printf.printf "[MUST ERROR: UNEXPECTED] %s\n" msg;
          false, ""
      end
  | TOR_error   ->
      begin
        try
          ignore (start debug [aopts1;aopts2] [] []);
          false, ""
        with IR.Must_error msg ->
          Printf.printf "[MUST ERROR: EXPECTED] %s\n" msg;
          true, "[Err]"
      end


let run_r
    (debug: bool) (* global debug switch --false: no debug, nowhere *)
    (aopts1: analysis_opts)
    (aopts2: analysis_opts)
    (oracle: test_oracle_r)
    : bool option (* validated or not ? *)
    * bool        (* expected outcome or not *) =
  try
    let b = start debug [aopts1; aopts2] [] [] in
    let o =
      match b, oracle with
      | true, TOR_succeed  -> true
      | true, TOR_fail     -> false
      | false, TOR_succeed -> false
      | false, TOR_fail    -> true
      | _, _               -> false in
    Some b, o
  with IR.Must_error msg ->
    match oracle with
    | TOR_error ->
        Printf.printf "[MUST ERROR: EXPECTED] %s\n" msg;
        None, true
    | TOR_succeed | TOR_fail ->
        Printf.printf "[MUST ERROR: UNEXPECTED] %s\n" msg;
        None, false
