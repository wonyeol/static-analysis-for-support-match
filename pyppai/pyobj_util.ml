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
 ** pyobj_util.ml: util for Py.Object.t *)
open Lib

       
(** global flag for Py.initialize *)
let pyml_initialized = ref false

(** construct ast *)
let get_ast (python_code: string): Py.Object.t =
  (* init Pyml *)
  if not !pyml_initialized then
    begin
      Py.initialize ~version:3 ();  (* use python 3 *)
      pyml_initialized := true
    end;
  (* set retval *)
  let m = Py.Import.add_module "ocaml" in
  let retval: Py.Object.t ref = ref Py.none in
  let callback (args: Py.Object.t array): Py.Object.t =
    retval := Array.get args 0;
    Py.none in
  Py.Module.set_function m "callback" callback;
  (* set py_str *)
  let py_str =
    let buf = Buffer.create 1 in
    Buffer.add_string buf "import ast\n";
    Buffer.add_string buf "from ocaml import callback\n";
    Buffer.add_string buf "e = '''";
    Buffer.add_string buf python_code;
    Buffer.add_string buf "'''\n";
    Buffer.add_string buf "callback(ast.parse(e))\n";
    Buffer.contents buf in
  (* run py_str *)
  ignore (Py.Run.simple_string py_str);
  (* return retval *)
  !retval


(** extract info *)
let get_attr (pyobj: Py.Object.t) (str: string): Py.Object.t =
  opt_get_fail (Py.Object.get_attr_string pyobj str)

let get_classname (pyobj: Py.Object.t): string =
  Py.String.to_string (get_attr (get_attr pyobj "__class__") "__name__")


(** print *)
let to_string (pyobj: Py.Object.t): string =
  let _pyobj_type (pyobj: Py.Object.t): Py.Object.t =
    Py.Object.get_type pyobj in
  let _pyobj_content (pyobj: Py.Object.t): Py.Object.t =
    if Py.Object.has_attr_string pyobj "__dict__" then
      get_attr pyobj "__dict__"
    else
      pyobj in
  let _pyobj_str (pyobj: Py.Object.t): string =
    Py.Object.to_string pyobj in
  (pyobj |> _pyobj_content |> _pyobj_str) ^ " : "
  ^ (pyobj |> _pyobj_type |> _pyobj_str)

let print pyobj = Printf.printf "%s" (to_string pyobj)
