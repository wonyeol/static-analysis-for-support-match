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
 ** entry_batch.ml: entry point for batch execution *)
open Ir_sig
open Lib
open Analysis_sig
open Main

module AF = Adom_fib

(** Types and general definitions *)

(* Debug information *)
let debug: bool ref = ref true

(** Results and display *)
type result_core = RPass | RFail | RError | RCrash | RTO
type result = result_core * string * float


(** Running jobs *)
exception Timeout
let run_job (i, tdn, aopts0, aopts1, oracle) acc =
  let get_prog_name aopts =
    match aopts.ao_input with
    | AI_pyfile s -> Printf.sprintf "file <%s>" s
    | AI_pystring s -> s in
  let input0 = get_prog_name aopts0 in
  let input1 = get_prog_name aopts1 in
  Printf.printf "#### TEST(R) %d ####\n" i;
  Printf.printf "entry0: %s\n" input0;
  Printf.printf "entry1: %s\n" input1;
  Printf.printf "expected result: %s\n\n" (string_of_test_oracle_r oracle);
  let time = ref 0. in
  let result, id =
    try
      let time_st = Unix.gettimeofday () in
      let out, is_ok = run_r !debug aopts0 aopts1 oracle in
      let time_ed = Unix.gettimeofday () in
      time := time_ed -. time_st;
      if is_ok then Printf.printf "[RESULT OF TEST(R) %d] PASSED\n\n" i
      else Printf.printf "[RESULT OF TEST(R) %d] FAILED\n\n" i;
      let r =
        match out with
        | Some b -> if b then RPass else RFail
        | None -> RError in
      r, is_ok
    with exn ->
      Printf.printf "[RESULT OF TEST(R) %d] CRASHED\n" i;
      Printf.printf "%s\n\n" (Printexc.to_string exn);
      if exn = Timeout then RTO,false
      else RCrash, false in
  IntMap.add i (id, result, oracle, !time, tdn) acc
let run_jobs_from_list timeout jobs acc =
  let rec aux jobs acc =
    match jobs with
    | [ ] -> acc
    | job :: others ->
        if !debug then
          Printf.printf "Job(s): remaining %d\n" (List.length jobs);
        if timeout > 0 then
          begin
            let handle_to _ =
              ignore (Unix.alarm timeout);
              raise Timeout in
            Sys.set_signal Sys.sigalrm (Sys.Signal_handle handle_to);
            ignore (Unix.alarm timeout)
          end;
        let acc = run_job job acc in
        aux others acc in
  aux jobs acc
let run_jobs timeout jobs =
  run_jobs_from_list timeout jobs IntMap.empty

(** Displaying results *)
let display_results results: unit =
  let comp m =
    IntMap.fold
      (fun _ (_, v,_,_,_) (tot, p, f, e, c, t) ->
        match v with
        | RPass  -> tot + 1, p + 1, f, e, c, t
        | RFail  -> tot + 1, p, f + 1, e, c, t
        | RError -> tot + 1, p, f, e + 1, c, t
        | RCrash -> tot + 1, p, f, e, c + 1, t
        | RTO    -> tot + 1, p, f, e, c, t + 1
      ) m (0, 0, 0, 0, 0, 0) in
  let pp_res_i i (id, res, exp, time, fname) =
    let sok = if id then "  yes" else "   no" in
    let exp =
      match exp with
      | TOR_succeed -> "  Valid"
      | TOR_fail    -> "Invalid"
      | TOR_error   -> "    ERR" in
    let res =
      match res with
      | RPass  -> "  Valid"
      | RFail  -> "Invalid"
      | RError -> "    ERR"
      | RCrash -> "  Crash"
      | RTO    -> "Timeout" in
    Printf.printf "%3d | %s | %s | %s | " i sok exp res;
    Printf.printf "  %.4f | %s\n" time fname in
  Printf.printf "\n\nRESULTS:\n\n";
  Printf.printf "Num | Match |  Expect |  Result | Time (s) | Test\n";
  Printf.printf
    "----+-------+---------+---------+----------+--------------------------\n";
  IntMap.iter pp_res_i results;
  Printf.printf
    "----+-------+---------+---------+----------+--------------------------\n";
  let count, rpass, rfail, rerror, rcrash, rtimeout = comp results in
  Printf.printf "\n\nSUMMARY:\n";
  Printf.printf " TESTS:   %d\n VALID:   %d\n INVALID: %d\n" count rpass rfail;
  Printf.printf " ERR:     %d\n CRASH:   %d\n TO:      %d\n" rerror rcrash
    rtimeout

(** Main function *)
let main () =
  let timeout: int ref = ref 20 (* optional timeout value; 0 is no timeout *) in
  (* No debug information *)
  let debug_off () =
    debug := false ;
    Ai_make.debug := false ;
    Main.output := false in
  (* All debug information *)
  let debug_on () =
    debug := false ;
    Ai_make.debug := false ;
    Main.output := false in
  (* Series of tests to be ran *)
  let series = ref "suite" in
  (* Parsing of arguments *)
  Arg.parse
    [ (* bench mode, turns out debug *)
      "-q",  Arg.Unit debug_off, "quiet mode, no debug information";
      "-v",  Arg.Unit debug_on , "verbose mode, debug information";
      (* sets timeout *)
      "-to", Arg.Int (fun i -> timeout := i), "sets timeout value (s)";
    ]
    (fun s -> series := s) "regression testing";
  (* Definition of the jobs to do *)
  let jobs =
    let tests =
      match !series with
      | "examples" -> Data.examples
      | "suite"    -> Data.suite
      | s -> failwith (Printf.sprintf "incorrect series of tests: %s" s) in
    let mk_rel i (a,b,c,d) = (i,a,b,c,d) in
    List.mapi mk_rel tests in
  (*job_list := jobs;*)
  let results = run_jobs !timeout jobs in
  display_results results

(* Start! *)
let _ = main ()
