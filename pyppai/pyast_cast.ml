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
 ** pyast_cast.ml: conversion from Py.Object.t to Pyast_sig *)
open Lib
open Pyast_sig
module Pyo = Pyobj_util
               
 
(** **************)
(** helper funcs *)
(** **************)
let pyobj_attr = Pyo.get_attr
let pyobj_classname = Pyo.get_classname
let pyobj_str = Pyo.to_string


(** *****************)
(** to_{primitives} *)
(** *****************)
let to_string: Py.Object.t -> string = Py.String.to_string
let to_int:    Py.Object.t -> int    = Py.Int.to_int
let to_float:  Py.Object.t -> float  = Py.Float.to_float
let to_bool:   Py.Object.t -> bool   = Py.Bool.to_bool
let to_list (f: Py.Object.t -> 'a) (pyobj: Py.Object.t): 'a list =
  Py.List.to_list_map f pyobj
let to_opt (f: Py.Object.t -> 'a) (pyobj: Py.Object.t): 'a option =
  match pyobj_classname pyobj with
  | "NoneType" -> None
  | _          -> Some (f pyobj)


(** **********************)
(** to_{pyast} (non-rec) *)
(** **********************)
let to_identifier: Py.Object.t -> identifier = to_string

let to_singleton: Py.Object.t -> singleton = to_opt to_bool

let to_number (pyobj: Py.Object.t): number =
  match pyobj_classname pyobj with
  | "int"   -> Int   (to_int   pyobj)
  | "float" -> Float (to_float pyobj)
  | _ -> failwith ("to_number: unreachable [" ^ (pyobj_str pyobj) ^ "]")

let to_expr_context (pyobj: Py.Object.t): expr_context =
  match pyobj_classname pyobj with
  | "Load"  -> Load
  | "Store" -> Store
  | "Del"   -> Del
  | "Param" -> Param
  | _  -> failwith ("to_expr_context: unreachable [" ^ (pyobj_str pyobj) ^ "]")

let to_boolop (pyobj: Py.Object.t): boolop =
  match pyobj_classname pyobj with
  | "And" -> And
  | "Or"  -> Or
  | _  -> failwith ("to_bool: unreachable [" ^ (pyobj_str pyobj) ^ "]")

let to_operator (pyobj: Py.Object.t): operator =
  match pyobj_classname pyobj with
  | "Add"         -> Add
  | "Sub"         -> Sub
  | "Mult"        -> Mult
  | "MatMult"     -> MatMult
  | "Div"         -> Div
  | "Mod"         -> Mod
  | "Pow"         -> Pow
  | "LShift"      -> LShift
  | "RShift"      -> RShift
  | "BitOr"       -> BitOr
  | "BitXor"      -> BitXor
  | "BitAnd"      -> BitAnd
  | "FloorDiv"    -> FloorDiv
  | _  -> failwith ("to_operator: unreachable [" ^ (pyobj_str pyobj) ^ "]")

let to_unaryop (pyobj: Py.Object.t): unaryop =
  match pyobj_classname pyobj with
  | "Invert"      -> Invert
  | "Not"         -> Not
  | "UAdd"        -> UAdd
  | "USub"        -> USub
  | _  -> failwith ("to_unaryop: unreachable [" ^ (pyobj_str pyobj) ^ "]")

let to_cmpop (pyobj: Py.Object.t): cmpop =
  match pyobj_classname pyobj with
  | "Eq"          -> Eq
  | "NotEq"       -> NotEq
  | "Lt"          -> Lt
  | "LtE"         -> LtE
  | "Gt"          -> Gt
  | "GtE"         -> GtE
  | "Is"          -> Is
  | "IsNot"       -> IsNot
  | "In"          -> In
  | "NotIn"       -> NotIn
  | _  -> failwith ("to_cmpop: unreachable [" ^ (pyobj_str pyobj) ^ "]")

                   
(** ******************)
(** to_{pyast} (rec) *)
(** ******************)
let rec to_modl (pyobj: Py.Object.t): 'a option modl =
  let a = None in
  match pyobj_classname pyobj with
  | "Module" -> Module (pyobj_attr pyobj "body" |> to_list to_stmt, a)
  | _ -> failwith ("to_modl: unreachable [" ^ (pyobj_str pyobj) ^ "]")

and to_stmt (pyobj: Py.Object.t): 'a option stmt =
  let a = None in
  match pyobj_classname pyobj with
  | "FunctionDef" ->
      FunctionDef (pyobj_attr pyobj "name"           |> to_identifier,
                   pyobj_attr pyobj "args"           |> to_arguments,
                   pyobj_attr pyobj "body"           |> to_list to_stmt,
                   pyobj_attr pyobj "decorator_list" |> to_list to_expr,
                   pyobj_attr pyobj "returns"        |> to_opt  to_expr, a)
  | "Return" ->
      Return (pyobj_attr pyobj "value"   |> to_opt  to_expr, a)
  | "Assign" ->
      Assign (pyobj_attr pyobj "targets" |> to_list to_expr,
              pyobj_attr pyobj "value"   |> to_expr, a)
  | "For"  ->
      For  (pyobj_attr pyobj "target" |> to_expr,
            pyobj_attr pyobj "iter"   |> to_expr,
            pyobj_attr pyobj "body"   |> to_list to_stmt,
            pyobj_attr pyobj "orelse" |> to_list to_stmt, a)
  | "While" ->
      While(pyobj_attr pyobj "test"   |> to_expr,
            pyobj_attr pyobj "body"   |> to_list to_stmt,
            pyobj_attr pyobj "orelse" |> to_list to_stmt, a)
  | "If"   ->
      If   (pyobj_attr pyobj "test"   |> to_expr,
            pyobj_attr pyobj "body"   |> to_list to_stmt,
            pyobj_attr pyobj "orelse" |> to_list to_stmt, a)
  | "With" ->
      With (pyobj_attr pyobj "items"  |> to_list to_withitem,
            pyobj_attr pyobj "body"   |> to_list to_stmt, a)
  | "Expr" ->
      Expr (pyobj_attr pyobj "value"  |> to_expr, a)
  | "Pass"     -> Pass     a
  | "Break"    -> Break    a
  | "Continue" -> Continue a
  | "ClassDef" ->
      (* hy: Ignore a class definition. This amounts to 
       * making an assumption that only global-side-effect free functions 
       * from the defined class are used. *)
      Pass a
  | _ -> 
      failwith ("to_stmt: unreachable [" ^ (pyobj_str pyobj) ^ "]")

and to_expr (pyobj: Py.Object.t): 'a option expr =
  let a = None in
  match pyobj_classname pyobj with
  | "BoolOp" ->
      BoolOp (pyobj_attr pyobj "op"      |> to_boolop,
              pyobj_attr pyobj "values"  |> to_list to_expr, a)
  | "BinOp" ->
      BinOp (pyobj_attr pyobj "left"    |> to_expr,
             pyobj_attr pyobj "op"      |> to_operator,
             pyobj_attr pyobj "right"   |> to_expr, a)
  | "UnaryOp" ->
      UnaryOp (pyobj_attr pyobj "op"      |> to_unaryop,
               pyobj_attr pyobj "operand" |> to_expr, a)
  | "Dict" ->
      Dict (pyobj_attr pyobj "keys"   |> to_list to_expr,
            pyobj_attr pyobj "values" |> to_list to_expr, a)
  | "Compare" ->
      Compare (pyobj_attr pyobj "left"        |> to_expr,
               pyobj_attr pyobj "ops"         |> to_list to_cmpop,
               pyobj_attr pyobj "comparators" |> to_list to_expr, a)
  | "Call" ->
      Call (pyobj_attr pyobj "func"     |> to_expr,
            pyobj_attr pyobj "args"     |> to_list to_expr,
            pyobj_attr pyobj "keywords" |> to_list to_keyword, a)
  | "Num"    ->
      Num (pyobj_attr pyobj "n"           |> to_number, a)
  | "Str"    ->
      Str (pyobj_attr pyobj "s"           |> to_string, a)
  (* | "FormattedValue" ->
        FormattedValue
          (pyobj_attr pyobj "value"       |> to_expr,
          pyobj_attr pyobj "conversion"  |> to_opt to_int,
          pyobj_attr pyobj "format_spec" |> to_opt to_expr, a) *)
  | "NameConstant" ->
      NameConstant (pyobj_attr pyobj "value" |> to_singleton, a)
  | "Attribute" ->
      Attribute (pyobj_attr pyobj "value" |> to_expr,
                 pyobj_attr pyobj "attr"  |> to_identifier,
                 pyobj_attr pyobj "ctx"   |> to_expr_context, a)
  | "Subscript" ->
      Subscript (pyobj_attr pyobj "value" |> to_expr,
                 pyobj_attr pyobj "slice"  |> to_slice,
                 pyobj_attr pyobj "ctx"   |> to_expr_context, a)
  | "Name"  ->
      Name (pyobj_attr pyobj "id"    |> to_identifier,
            pyobj_attr pyobj "ctx"   |> to_expr_context, a)
  | "List"  ->
      List (pyobj_attr pyobj "elts"  |> to_list to_expr,
            pyobj_attr pyobj "ctx"   |> to_expr_context, a)
  | "Tuple" ->
      Tuple (pyobj_attr pyobj "elts"  |> to_list to_expr,
             pyobj_attr pyobj "ctx"   |> to_expr_context, a)
  | _ -> failwith ("to_expr: unreachable [" ^ (pyobj_str pyobj) ^ "]")

and to_slice (pyobj: Py.Object.t): 'a option slice = 
  match pyobj_classname pyobj with
  | "Slice" -> 
      Slice    (pyobj_attr pyobj "lower" |> to_opt to_expr,
                pyobj_attr pyobj "upper" |> to_opt to_expr,
                pyobj_attr pyobj "step"  |> to_opt to_expr)
  | "ExtSlice" ->
      ExtSlice (pyobj_attr pyobj "dims"  |> to_list to_slice)
  | "Index" -> 
      Index    (pyobj_attr pyobj "value" |> to_expr)
  | _ -> failwith ("to_slice: unreachable [" ^ (pyobj_str pyobj) ^ "]")

and to_arguments (pyobj: Py.Object.t): 'a option arguments =
  (pyobj_attr pyobj "args"        |> to_list to_expr,
   pyobj_attr pyobj "vararg"      |> to_opt  to_arg,
   pyobj_attr pyobj "kwonlyargs"  |> to_list to_arg,
   pyobj_attr pyobj "kw_defaults" |> to_list to_expr,
   pyobj_attr pyobj "kwargs"      |> to_opt  to_arg,
   pyobj_attr pyobj "defaults"    |> to_list to_expr)

and to_arg (pyobj: Py.Object.t): 'a option arg =
  let a = None in
  (pyobj_attr pyobj "arg"        |> to_identifier,
   pyobj_attr pyobj "annotation" |> to_opt to_expr, a)

and to_keyword (pyobj: Py.Object.t): 'a option keyword =
  (pyobj_attr pyobj "arg"   |> to_opt to_identifier,
   pyobj_attr pyobj "value" |> to_expr)
    
and to_withitem (pyobj: Py.Object.t): 'a option withitem =
  (pyobj_attr pyobj "context_expr"  |> to_expr,
   pyobj_attr pyobj "optional_vars" |> to_opt to_expr)

let pyobj_to_modl = to_modl
