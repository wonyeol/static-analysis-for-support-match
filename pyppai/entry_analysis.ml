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
 ** entry_analysis.ml: entry file for the command line analysis *)
open Analysis_sig

module IR = Ir_sig

(* Main function:
 * 1. parses argument
 * 2. calls start_nr with appropriate arguments
 *)
let main () =
  (* debug information *)
  let debug = ref true in
  let debug_off () =
    debug := false ;
    Ai_make.debug := false ;
    Main.output := false in
  let debug_on () =
    debug := false ;
    Ai_make.debug := false ;
    Main.output := false in
  (* python input file *)
  let file = ref None in
  (* python strings *)
  let ix_ps = ref 0 in
  let fset_ps i = ix_ps := i in
  (* kinds of analysis *)
  let ai_box  = ref false
  and ai_oct  = ref false
  and ai_pol  = ref false in
  (* analysis settings *)
  let do_thr  = ref false in
  let do_sim_assm = ref false in
  (* zone domain *)
  let do_zone = ref false in
  (* model and guide pair *)
  let model = ref None
  and guide = ref None in
  let fopt r s = r := Some s in
  (* parsing command line arguments *)
  Arg.parse
    [ "-ai-box",         Arg.Set   ai_box,        "Num analysis, Boxes" ;
      "-ai-oct",         Arg.Set   ai_oct,        "Num analysis, Octagons" ;
      "-ai-pol",         Arg.Set   ai_pol,        "Num analysis, Polyhedra" ;
      "-set-ps",         Arg.Int   fset_ps,       "Sets python test string" ;
      "-thr-on",         Arg.Set   do_thr,        "Threshold widening ON" ;
      "-thr-off",        Arg.Clear do_thr,        "Threshold widening OFF" ;
      "-zone-on",        Arg.Set   do_zone,       "Zone domain ON" ;
      "-zone-off",       Arg.Clear do_zone,       "Zone domain OFF" ;
      "-sim-assm-on",    Arg.Set   do_sim_assm,
      "Expression simplfication in Assume ON" ;
      "-sim-assm-off",   Arg.Clear do_sim_assm,
      "Expression simplification in Assume OFF" ;
      (* setting model and guide *)
      "-model",          Arg.String (fopt model), "Sets the model";
      "-guide",          Arg.String (fopt guide), "Sets the guide";
      (* setting the verbosity level *)
      "-q",              Arg.Clear debug, "quiet mode, no debug information";
      "-v",              Arg.Set   debug , "verbose mode, debug information";
    ]
    (fun s -> file := Some s)
    "pyppai, a basic python probabilistic programs abstract interpter";
  if !debug then debug_on () else debug_off ();
  let do_num =
    if !ai_pol then Some AD_pol
    else if !ai_oct then Some AD_oct
    else if !ai_box then Some AD_box
    else None in
  match !model, !guide with
  | None, None ->
      let input =
        match !file with
        | None -> failwith "no input file given"
        | Some s -> AI_pyfile s in
      let b =
        Main.start_nr false { ao_do_num   = do_num;
                              ao_debug_it = !debug;
                              ao_input    = input;
                              ao_wid_thr  = !do_thr; 
                              ao_sim_assm = !do_sim_assm;
                              ao_zone     = !do_zone } 
          [IR.True]
          [IR.False] in
      let s = if b then "VALID" else "POSSIBLY INVALID" in
      Printf.printf "\nAnalysis result: program %s\n\n" s
  | Some m, Some g ->
      let aopts1 = { ao_do_num   = do_num;
                     ao_debug_it = !debug;
                     ao_input    = AI_pyfile m;
                     ao_wid_thr  = !do_thr;
                     ao_sim_assm = !do_sim_assm;
                     ao_zone     = !do_zone } in
      let aopts2 = { aopts1 with ao_input = AI_pyfile g } in
      let b = Main.start !debug [ aopts1; aopts2 ] [] [] in
      let s = if b then "VALID" else "POSSIBLY INVALID" in
      Printf.printf "\nAnalysis result: model guide pair %s\n\n" s
  | Some _, None | None, Some _ ->
      failwith "inconsistent mode; either give model and guide or give neither"

(* Start! *)
let _ = ignore (main ())
