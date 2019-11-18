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
 ** lib.ml: basic data-types and utilities *)

module F = Format

(** Data structures *)
module StringSet = Set.Make(String)
module StringMap = Map.Make(String)
module SS = StringSet
module SM = StringMap

module Int =
  struct
    type t = int
    let compare n1 n2 = n1 - n2
  end
module IntSet = Set.Make(Int)
module IntMap = Map.Make(Int)
module IM = IntMap

(** Name manager *)
module NameManager : sig
  val reset : unit -> unit
  val create_name : unit -> string
  val return_name : string
  val plate_indices_name : string -> string
  (* plate_indices_name:
   *   `plate_indices_name(p_name)' denotes the field `_indices' of p_name,
   *   i.e., `p_name._indices' in python, which stores the values of subsampled indices. *)
  (* val plate_dim_name : string -> string
  (* plate_dim_name:
   *   `plate_dim_name(p_name)' denotes `p_name.dim' in python.*) *)
end = struct
  let counter = ref (-1)
  let base_name = "__@@temp"
  let reset () = counter := -1
  let create_name () : string =
    counter := !counter + 1;
    base_name ^ (string_of_int (!counter))
  let return_name : string = "__@@ret"
  let plate_indices_name s = Printf.sprintf "__@@_indices(%s)" s
  (* let plate_dim_name s = Printf.sprintf "__@@dim(%s)" s *)
end

(** for printer: Conversion of printer functions *)
let buf_to_string (f: Buffer.t -> 'a -> unit) (x: 'a): string =
  let buf = Buffer.create 1 in
  f buf x;
  Buffer.contents buf
let buf_to_channel (f: Buffer.t -> 'a -> unit)
    (chan: out_channel) (x: 'a): unit =
  let buf = Buffer.create 1 in
  f buf x;
  Buffer.output_buffer chan buf

(** Some generic printing functions *)
(* (maybe not used for now, but try to have more standard interfaces *)
let buf_int (buf: Buffer.t) (i: int): unit = Printf.bprintf buf "%d" i
let buf_opt (buf_a: Buffer.t -> 'a -> unit) (buf: Buffer.t) (opt: 'a option)
    : unit =
  match opt with
  | None -> Printf.bprintf buf "None"
  | Some a -> Printf.bprintf buf "Some %a" buf_a a
let buf_list (to_str: 'a -> string) (sep: string) (buf: Buffer.t)
    (l: 'a list): string =
  match l with
  | [ ] -> ""
  | [ n ] -> to_str n
  | n :: ns -> 
      let buf = Buffer.create 1 in
      Printf.bprintf buf "%s" (to_str n);
      List.iter (fun n0 -> Printf.bprintf buf "%s" (sep ^ to_str n0)) ns;
      Buffer.contents buf


(** for printer: set *)
(* xr: we can also extend the map module to make this more systematic,
 * for all types *)
let buf_ss (buf: Buffer.t) (x: SS.t): unit =
  Printf.bprintf buf "{ ";
  SS.iter (Printf.bprintf buf "%s; ") x;
  Printf.bprintf buf "}"
let ss_to_string = buf_to_string buf_ss
let ss_pp = buf_to_channel buf_ss

(** to string *)
let opt_to_string (to_str: 'a -> string) = function
  | None -> "?"
  | Some(v) -> to_str v

let int_opt_to_string (v: int option): string =
  opt_to_string string_of_int v

let string_opt_to_string (v: string option): string =
  opt_to_string (fun x -> x) v

let list_to_string (to_str: 'a -> string) (sep: string)
    : 'a list -> string = function
  | [] -> ""
  | [n] -> to_str n
  | n :: ns -> 
      let buf = Buffer.create 1 in
      Printf.bprintf buf "%s" (to_str n);
      List.iter (fun n0 -> Printf.bprintf buf "%s" (sep ^ to_str n0)) ns;
      Buffer.contents buf

let smap_to_string (to_str: 'a -> string) (sep: string) (m: 'a SM.t) = 
  if (SM.cardinal m) = 0 then ""
  else 
    let first_time = ref true in 
    let buf = Buffer.create 1 in
    let f k v = 
      if !first_time = false then
        Printf.bprintf buf "%s" (sep ^ k ^ "->" ^ to_str v)
      else 
        begin
          Printf.bprintf buf "%s" (sep ^ k ^ "->" ^ to_str v);
          first_time := false
        end in
    SM.iter f m;
    Buffer.contents buf

(** for file *)
let read_file (fname : string) : string =
  let file = Unix.openfile fname [ Unix.O_RDONLY ] 0o644 in
  let inchan = Unix.in_channel_of_descr file in
  let buf = Buffer.create 1 in
  begin
    try
      while true do
        Buffer.add_string buf ((input_line inchan) ^ "\n")
      done
    with End_of_file -> ( )
  end;
  Buffer.contents buf

(** for option *)
let opt_get_fail: 'a option -> 'a = function
  | Some v -> v
  | None   -> failwith "opt_get: None"

let opt_get_default (default: 'a): 'a option -> 'a = function
  | Some v -> v
  | None   -> default

let opt_some_eq (v1: 'a option) (v2: 'a option) =
  match v1, v2 with
  | Some v1, Some v2 -> v1 = v2
  | _ -> false

let opt_some_leq (v1: 'a option) (v2: 'a option) =
  match v1, v2 with
  | Some v1, Some v2 -> v1 <= v2
  | _ -> false

let lift_opt (f: 'a -> 'b option) : 'a option -> 'b option = function
  | None -> None
  | Some x -> f x

(** for string *)
let is_suffix str1 str2 =
  let len1 = String.length str1 in
  let len2 = String.length str2 in
  (len2 <= len1) && ((String.sub str1 (len1-len2) len2) = str2)

(** for set *)
(* TODO: [xr] I see no reason not to use SS.equal;
 * indeed, it denotes semantic equality in OCaml *)
let are_same_sets (s1: SS.t) (s2: SS.t): bool =
  SS.subset s1 s2 && SS.subset s2 s1

(** for map *)
let merge_smap (m1: 'a SM.t) (m2: 'b SM.t): ('a * 'b) SM.t =
  let merge k aopt1 bopt2 =
    match aopt1, bopt2 with
    | Some a1, Some b2 -> Some (a1, b2)
    | _ -> None in
  SM.merge merge m1 m2

let merge_imap (m1: 'a IM.t) (m2: 'b IM.t): ('a * 'b) IM.t =
  let merge k aopt1 bopt2 =
    match aopt1, bopt2 with
    | Some a1, Some b2 -> Some (a1, b2)
    | _ -> None in
  IM.merge merge m1 m2

(** for list *)
(* Let l = [l0; l1; ...; l(N-1)]. *)
(* list_take n l = [l0; l1; ...; l(n-1)] *)
let list_take (n: int) (l: 'a list): 'a list =
  let rec f (acc: 'a list) (n0: int) (l0: 'a list): 'a list =
    if n0 <= 0 then List.rev acc
    else
      match l0 with
      | [] -> List.rev acc
      | x1::l1 -> f (x1::acc) (n0 - 1) l1 in
  f [] n l

(* list_drop n l = [ln; l(n+1); ...; l(N-1)] *)
let rec list_drop (n: int) (l: 'a list): 'a list =
  if n <= 0 then l
  else
    match l with
    | [] -> []
    | _::l0 -> list_drop (n-1) l0

(* list_take_last n l = [l(N-n); l(N-n+1); ...; l(N-1)] *)
let list_take_last (n: int) (l: 'a list): 'a list =
  List.rev (list_take n (List.rev l))

(* list_drop_last n l = [l0; l1; ...; l(N-1-n)] *)
let list_drop_last (n: int) (l: 'a list): 'a list =
  List.rev (list_drop n (List.rev l))

let buf_list (sep: string) (buf_elt: Buffer.t -> 'a -> unit)
    (buf: Buffer.t) (l: 'a list): unit =
  let r = ref false in
  List.iter
    (fun e ->
      Printf.bprintf buf "%s%a" (if !r then sep else "") buf_elt e;
      r := true
    ) l

(* list_repeat n x = [x]*n *)  
let list_repeat (n: int) (x: 'a) =
  let rec f (acc: 'a list) (n0: int) =
    if n0 <= 0 then acc else f (x::acc) (n0-1) in
  f [] n

let lift_to_list (convert: 'a -> 'b list * 'c) (l: 'a list):
      'b list * 'c list  =
  let converted_l = List.map convert l in
  (List.flatten (List.map fst converted_l), List.map snd converted_l)

(* list_replace n x l = [l0; ...; l(n-1); x; l(n+1); ...; l(N-1)] *)
let rec list_replace (n: int) (x: 'a) (l: 'a list): 'a list =
  match l with
  | [] -> []
  | hd :: tl ->
     if n=0 then x :: tl
     else hd :: (list_replace (n-1) x tl)

(* list_drop_leading_elt x [x; x; ...; x; y; ...] = [y; ...] when x != y *)
let rec list_drop_leading_elt (x: 'a) (l: 'a list): 'a list =
  match l with
  | [] -> []
  | hd :: tl ->
     let hd_new = if hd = x then [] else [hd] in
     hd_new @ (list_drop_leading_elt x tl)
  
(* list_drop_trailing_elt x [...; y; x; x; ...; x] = [...; y] when x != y *)
let list_drop_trailing_elt (x: 'a) (l: 'a list): 'a list =
  List.rev (list_drop_leading_elt x (List.rev l))

(* list_find_index x (l1 @ [x] @ l2) = len(l1), where x \notin l1.
   list_find_index x l = -1, where x \notin l. *)
let list_find_index (x: 'a) (l: 'a list): int =
  let rec f (cur_i: int) (cur_l: 'a list): int =
    match cur_l with
    | [] -> (-1)
    | hd :: tl -> if x = hd then cur_i else f (cur_i+1) tl in
  f 0 l
