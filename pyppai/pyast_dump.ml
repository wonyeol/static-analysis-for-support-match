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
 ** pyast_dump.ml: dumping Python abstract syntax trees
 **
 ** This file is based on dump.ml in ocaml-pythonlib:
 **   https://github.com/m2ym/ocaml-pythonlib/blob/master/src/dump.ml
 **)
open Pyast_sig
open Pyast_util
open Lib

module F = Format


(** Printing basic types *)
let pp_char   = F.pp_print_char
let pp_string = F.pp_print_string
let pp_int    = F.pp_print_int
let pp_float  = F.pp_print_float
let pp_bool   = F.pp_print_bool
let pp_list ?(bracket=true) pp fmt list =
  let rec loop fmt = function
    | [] -> ()
    | [x] -> pp fmt x
    | x :: xs -> F.fprintf fmt "%a, %a" pp x loop xs in
  if bracket then
    F.fprintf fmt "[%a]" loop list
  else
    loop fmt list
let pp_opt pp fmt = function
  | None -> pp_string fmt "None"
  | Some x -> pp fmt x 


(** Printing to formatters *)

(* pp_{built-in types} *)
let pp_identifier fmt id = F.fprintf fmt "'%s'" id
let pp_num fmt num = pp_string fmt (string_of_number num)
let pp_str fmt str =
  (* TODO *)
  let escape_char d fmt = function
    | '\r' -> pp_string fmt "\\r"
    | '\n' -> pp_string fmt "\\n"
    | '\\' -> pp_string fmt "\\\\"
    | '"' when d -> pp_string fmt "\\\""
    | '\'' when not d -> pp_string fmt "\\'"
    | c -> pp_char fmt c in
  let escape_str d fmt s =
    String.iter (escape_char d fmt) s in
  if String.contains str '\'' && not (String.contains str '"') then
    F.fprintf fmt "\"%a\"" (escape_str true) str
  else
    F.fprintf fmt "\'%a\'" (escape_str false) str

(* pp_node *)
let rec pp_node fmt label fields =
  F.fprintf fmt "%s(%a)" label (pp_list ~bracket:false pp_field) fields

and pp_field fmt (label, field) =
  F.fprintf fmt "%s=" label;
  match field with
  | `Identifier      id  ->         pp_identifier fmt id
  | `Identifier_opt  id  -> pp_opt  pp_identifier fmt id
  | `Identifier_list ids -> pp_list pp_identifier fmt ids
  | `Num num -> pp_num fmt num
  | `Str str -> pp_str fmt str
  | `Int_opt i -> pp_opt pp_int fmt i
  | `NameConstant b  -> pp_opt pp_bool fmt b
  | `Stmt      stmt  ->         pp_stmt fmt stmt
  | `Stmt_list stmts -> pp_list pp_stmt fmt stmts
  | `Expr      expr  ->         pp_expr fmt expr
  | `Expr_opt  expr  -> pp_opt  pp_expr fmt expr
  | `Expr_list exprs -> pp_list pp_expr fmt exprs
  | `Expr_context ctx -> pp_expr_context fmt ctx
  | `Slice slice -> pp_slice fmt slice 
  | `Slice_list slices -> pp_list pp_slice fmt slices
  | `Boolop     op  ->         pp_boolop   fmt op
  | `Operator   op  ->         pp_operator fmt op
  | `Unaryop    op  ->         pp_unaryop  fmt op
  | `Cmpop      op  ->         pp_cmpop    fmt op
  | `Cmpop_list ops -> pp_list pp_cmpop    fmt ops
  (* | `Comprehension_list comps -> pp_list pp_comprehension fmt comps *)
  (* | `Excepthandler_list handlers -> pp_list pp_excepthandler fmt handlers *)
  | `Arguments args ->         pp_arguments fmt args
  | `Arg_opt   arg  -> pp_opt  pp_arg fmt arg
  | `Arg_list  args -> pp_list pp_arg fmt args
  | `Keyword_list kwds  -> pp_list pp_keyword  fmt kwds
  (* | `Alias_list aliases -> pp_list pp_alias fmt aliases *)
  | `Withitem_list wits -> pp_list pp_withitem fmt wits

(* pp_{module Python} *)
and pp_modl fmt v_modl =
  let cname = name_of_modl v_modl in
  match v_modl with
  | Module (body, _) ->
      pp_node fmt cname ["body", `Stmt_list body]

  (* | Interactive (body, _) ->
      pp_node fmt cname ["body", `Stmt_list body]
  | Expression (body, _) ->
      pp_node fmt cname ["body", `Expr body]
  | Suite (body, _) ->
      pp_node fmt cname ["body", `Stmt_list body] *)

and pp_stmt fmt v_stmt =
  let cname = name_of_stmt v_stmt in
  match v_stmt with
  | FunctionDef (name, args, body, decorator_list, returns, _) ->
      pp_node fmt cname
        ["name", `Identifier name;
         "args", `Arguments args;
         "body", `Stmt_list body;
         "decorator_list", `Expr_list decorator_list;
         "returns", `Expr_opt returns]

  (* | ClassDef (name, bases, body, decorator_list, _) ->
      pp_node fmt cname
        ["name", `Identifier name;
         "bases", `Expr_list bases;
         "body", `Stmt_list body;
         "decorator_list", `Expr_list decorator_list] *)

  | Return (value, _) ->
      pp_node fmt cname ["value", `Expr_opt value]

  (* | Delete (targets, _) ->
      pp_node fmt cname ["targets", `Expr_list targets] *)

  | Assign (targets, value, _) ->
      pp_node fmt cname
        ["targets", `Expr_list targets;
         "value", `Expr value]

  (* | AugAssign (target, op, value, _) ->
      pp_node fmt cname
        ["target", `Expr target;
         "op", `Operator op;
         "value", `Expr value] *)

  | For (target, iter, body, orelse, _) ->
      pp_node fmt cname
        ["target", `Expr target;
         "iter", `Expr iter;
         "body", `Stmt_list body;
         "orelse", `Stmt_list orelse]
  | While (test, body, orelse, _) ->
      pp_node fmt cname
        ["test", `Expr test;
         "body", `Stmt_list body;
         "orelse", `Stmt_list orelse]
  | If (test, body, orelse, _) ->
      pp_node fmt cname
        ["test", `Expr test;
         "body", `Stmt_list body;
         "orelse", `Stmt_list orelse]
  | With (items, body, _) ->
      pp_node fmt cname
        ["items", `Withitem_list items;
         "body", `Stmt_list body]

  (* | Raise (typ, inst, tback, _) ->
      pp_node fmt cname
        ["type", `Expr_opt typ;
         "inst", `Expr_opt inst;
         "tback", `Expr_opt tback]
  | Try (body, handlers, orelse, _) ->
      pp_node fmt cname
        ["body", `Stmt_list body;
         "handlers", `Excepthandler_list handlers;
         "orelse", `Stmt_list orelse]
  | Assert (test, msg, _) ->
      pp_node fmt cname
        ["test", `Expr test;
         "msg", `Expr_opt msg]
  | Import (names, _) ->
      pp_node fmt cname ["names", `Alias_list names]
  | ImportFrom (modul, names, level, _) ->
      (match level with
       | Some l ->
           pp_node fmt cname
             ["module", `Identifier modul;
              "names", `Alias_list names;
              "level", `Int l]
       | None -> failwith "Unreachable")
  | Global (names, _) ->
      pp_node fmt cname ["names", `Identifier_list names] *)

  | Expr (value, _) -> pp_node fmt cname ["value", `Expr value]
  | Pass (_) -> pp_node fmt cname []
  | Break (_) -> pp_node fmt cname []
  | Continue (_) -> pp_node fmt cname []

and pp_expr fmt v_expr =
  let cname = name_of_expr v_expr in
  match v_expr with
  | BoolOp (op, values, _) ->
      pp_node fmt cname
        ["op", `Boolop op;
         "values", `Expr_list values]
  | BinOp (left, op, right, _) ->
      pp_node fmt cname
        ["left", `Expr left;
         "op", `Operator op;
         "right", `Expr right]
  | UnaryOp (op, operand, _) ->
      pp_node fmt cname
        ["op", `Unaryop op;
         "operand", `Expr operand]

  (* | Lambda (args, body, _) ->
      pp_node fmt cname
        ["args", `Arguments args;
         "body", `Expr body]
  | IfExp (test, body, orelse, _) ->
      pp_node fmt cname
        ["test", `Expr test;
         "body", `Expr body;
         "orelse", `Expr orelse;] *)
  | Dict (keys, values, _) ->
      pp_node fmt cname
        ["keys", `Expr_list keys;
         "values", `Expr_list values]
  (* | ListComp (elt, generators, _) ->
      pp_node fmt cname
        ["elt", `Expr elt;
         "generators", `Comprehension_list generators]
  | GeneratorExp (elt, generators, _) ->
      pp_node fmt cname
        ["elt", `Expr elt;
         "generators", `Comprehension_list generators]
  | Yield (value, _) ->
      pp_node fmt cname ["value", `Expr_opt value] *)

  | Compare (left, ops, comparators, _) ->
      pp_node fmt cname
        ["left", `Expr left;
         "ops", `Cmpop_list ops;
         "comparators", `Expr_list comparators]

  | Call (func, args, keywords, _) ->
      pp_node fmt cname
        ["func", `Expr func;
         "args", `Expr_list args;
         "keywords", `Keyword_list keywords]

  | Num (n, _) -> pp_node fmt cname ["n", `Num n]

  | Str (s, _) -> pp_node fmt cname ["s", `Str s]
              
  (* | FormattedValue (value, conversion, format_spec, _) ->
     pp_node fmt cname
       ["value", `Expr value;
        "conversion", `Int_opt conversion;
        "format_spec", `Expr_opt format_spec] *)
              
  | NameConstant (value, _) -> pp_node fmt cname ["value", `NameConstant value]
              
  | Attribute (value, attr, ctx, _) ->
      pp_node fmt cname
        ["value", `Expr value;
         "attr", `Identifier attr;
         "ctx", `Expr_context ctx]

  | Subscript (value, slice, ctx, _) ->
      pp_node fmt cname
        ["value", `Expr value;
         "slice", `Slice slice;
         "ctx", `Expr_context ctx]

  | Name (id, ctx, _) ->
      pp_node fmt cname
        ["id", `Identifier id;
         "ctx", `Expr_context ctx]

  | List (elts, ctx, _) ->
      pp_node fmt cname
        ["elts", `Expr_list elts;
         "ctx", `Expr_context ctx]

  | Tuple (elts, ctx, _) ->
      pp_node fmt cname
        ["elts", `Expr_list elts;
         "ctx", `Expr_context ctx]

and pp_expr_context fmt ctx = pp_node fmt (name_of_expr_context ctx) []

and pp_slice fmt = function
  | Slice (lower, upper, step) ->
      pp_node fmt "Slice"
        ["lower", `Expr_opt lower;
         "upper", `Expr_opt upper;
         "step", `Expr_opt step]
  | ExtSlice (dims) ->
      pp_node fmt "ExtSlice" ["dims", `Slice_list dims]
  | Index (value) ->
      pp_node fmt "Index" ["value", `Expr value] 

and pp_boolop fmt op = pp_node fmt (name_of_boolop op) []

and pp_operator fmt op = pp_node fmt (name_of_operator op) []

and pp_unaryop fmt op = pp_node fmt (name_of_unaryop op) []

and pp_cmpop fmt op = pp_node fmt (name_of_cmpop op) []

(* and pp_comprehension fmt (target, iter, ifs) =
  pp_node fmt "comprehension"
    ["target", `Expr target;
     "iter", `Expr iter;
     "ifs", `Expr_list ifs] *)

(* and pp_excepthandler fmt = function
  | ExceptHandler (typ, name, body, pos) ->
      pp_node fmt "ExceptHandler"
        ["type", `Expr_opt typ;
         "name", `Expr_opt name;
         "body", `Stmt_list body] *)

and pp_arguments fmt (args, vararg, kwonlyargs, kw_defaults, kwarg, defaults) =
  pp_node fmt "arguments"
    ["args", `Expr_list args;
     "vararg", `Arg_opt vararg;
     "kwonlyargs", `Arg_list kwonlyargs;
     "kw_defaults", `Expr_list kw_defaults;
     "kwarg", `Arg_opt kwarg;
     "defaults", `Expr_list defaults]

and pp_arg fmt (arg, annotation, _) =
  pp_node fmt "arg"
    ["identifier", `Identifier arg;
     "annotation", `Expr_opt annotation]

and pp_keyword fmt (arg, value) =
  pp_node fmt "keyword"
    ["arg", `Identifier_opt arg;
     "value", `Expr value]

(* and pp_alias fmt (name, asname) =
  pp_node fmt "alias"
    ["name", `Identifier name;
     "asname", `Identifier_opt asname] *)

and pp_withitem fmt (context_expr, optional_vars) =
  pp_node fmt "withitem"
    ["context_expr", `Expr context_expr;
     "optional_vars", `Expr_opt optional_vars]
    

(** Printing to stdout *)
let pp_print_modl = pp_modl
let pp_print_stmt = pp_stmt
let pp_print_expr = pp_expr
let print_modl modl = pp_print_modl F.std_formatter modl
let print_stmt stmt = pp_print_stmt F.std_formatter stmt
let print_expr expr = pp_print_expr F.std_formatter expr


(** Generation of strings *)
let print_to_string pp node =
  let buf = Buffer.create 4096 in
  let fmt = F.formatter_of_buffer buf in
  pp fmt node;
  F.pp_print_flush fmt ();
  Buffer.contents buf
let dump_modl modl = print_to_string pp_print_modl modl
let dump_stmt stmt = print_to_string pp_print_stmt stmt
let dump_expr expr = print_to_string pp_print_expr expr
