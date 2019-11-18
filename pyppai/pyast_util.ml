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
 ** pyast_util.ml: utilities on Python AST *)
open Pyast_sig


(** Extraction of names *)
let name_of_number = function
  | Int _       -> "Int"
  | Float _     -> "Float"
  (* | LongInt _   -> "LongInt" *)
  (* | Imag _      -> "Imag" *)
and name_of_modl = function
  | Module _      -> "Module"
  (* | Interactive _ -> "Interactive" *)
  (* | Expression _  -> "Expression" *)
  (* | Suite _       -> "Suite" *)
and name_of_stmt = function
  | FunctionDef _ -> "FunctionDef"
  (* | AsyncFunctionDef _ -> ... *)
  (* | ClassDef _    -> "ClassDef" *)
  | Return _      -> "Return"
  (* | Delete _      -> "Delete" *)
  | Assign _      -> "Assign"
  (* | AugAssign _   -> "AugAssign" *)
  (* | AnnAssign _   -> ... *)
  | For _         -> "For"
  (* | AsyncFor _    -> ... *)
  | While _       -> "While"
  | If _          -> "If"
  | With _        -> "With"
  (* | AsyncWith _   -> ... *)
  (* | Raise _       -> "Raise" *)
  (* | Try _         -> "Try" *)
  (* | Assert _      -> "Assert" *)
  (* | Import _      -> "Import" *)
  (* | ImportFrom _  -> "ImportFrom" *)
  (* | Global _      -> "Global" *)
  (* | Nonlocal _    -> ... *)
  | Expr _        -> "Expr"
  | Pass _        -> "Pass"
  | Break _       -> "Break"
  | Continue _    -> "Continue"
and name_of_expr = function
  | BoolOp _       -> "BoolOp"
  | BinOp _        -> "BinOp"
  | UnaryOp _      -> "UnaryOp"
  (* | Lambda _       -> "Lambda" *)
  (* | IfExp _        -> "IfExp" *)
  | Dict _         -> "Dict"
  (* | ListComp _     -> "ListComp" *)
  (* | SetComp _      -> ... *)
  (* | DictComp _     -> ... *)
  (* | GeneratorExp _ -> "GeneratorExp" *)
  (* | Await _        -> ... *)
  (* | Yield _        -> "Yield" *)
  (* | YieldFrom _    -> "YieldFrom" *)
  | Compare _      -> "Compare"
  | Call _         -> "Call"
  | Num _          -> "Num"
  | Str _          -> "Str"
  (* | FormattedValue _ -> "FormattedValue" *)
  (* | JoinedStr _    -> ... *)
  (* | Bytes _        -> ... *)
  | NameConstant _ -> "NameConstant"
  (* | Ellipsis _     -> ... *)
  (* | Constant _     -> ... *)
  | Attribute _    -> "Attribute"
  | Subscript _    -> "Subscript" 
  (* | Starred _      -> ... *)
  | Name _         -> "Name"
  | List _         -> "List"
  | Tuple _        -> "Tuple"
and name_of_expr_context = function
  | Load        -> "Load"
  | Store       -> "Store"
  | Del         -> "Del"
  (* | AugLoad     -> "AugLoad" *)
  (* | AugStore    -> "AugStore" *)
  | Param       -> "Param"
(* and name_of_slice = function
  | Slice _     -> "Slice"
  | ExtSlice _  -> "ExtSlice"
  | Index _     -> "Index" *)
and name_of_boolop = function
  | And -> "And"
  | Or  -> "Or"
and name_of_operator = function
  | Add         -> "Add"
  | Sub         -> "Sub"
  | Mult        -> "Mult"
  | MatMult     -> "MatMult"
  | Div         -> "Div"
  | Mod         -> "Mod"
  | Pow         -> "Pow"
  | LShift      -> "LShift"
  | RShift      -> "RShift"
  | BitOr       -> "BitOr"
  | BitXor      -> "BitXor"
  | BitAnd      -> "BitAnd"
  | FloorDiv    -> "FloorDiv"
and name_of_unaryop = function
  | Invert      -> "Invert"
  | Not         -> "Not"
  | UAdd        -> "UAdd"
  | USub        -> "USub"
and name_of_cmpop = function
  | Eq          -> "Eq"
  | NotEq       -> "NotEq"
  | Lt          -> "Lt"
  | LtE         -> "LtE"
  | Gt          -> "Gt"
  | GtE         -> "GtE"
  | Is          -> "Is"
  | IsNot       -> "IsNot"
  | In          -> "In"
  | NotIn       -> "NotIn"
(* and name_of_excepthandler = function
  | ExceptHandler _ -> "ExceptHandler" *)


(** Conversion to strings *)
let string_of_number = function
  | Int (n)      -> string_of_int n
  | Float (n)    -> string_of_float n
  (* | LongInt (n)  -> (string_of_int n) ^ "L" *)
  (* | Imag (n)     -> n *)
let string_of_boolop = function
  | And -> "and"
  | Or  -> "or"
let string_of_operator = function
  | Add         -> "+"
  | Sub         -> "-"
  | Mult        -> "*"
  | MatMult     -> "@"
  | Div         -> "/"
  | Mod         -> "%"
  | Pow         -> "**"
  | LShift      -> "<<"
  | RShift      -> ">>"
  | BitOr       -> "|"
  | BitXor      -> "^"
  | BitAnd      -> "&"
  | FloorDiv    -> "//"
let string_of_unaryop = function
  | Invert -> "~"
  | Not    -> "not"
  | UAdd   -> "+"
  | USub   -> "-"
let string_of_cmpop = function
  | Eq    -> "=="
  | NotEq -> "!="
  | Lt    -> "<"
  | LtE   -> "<="
  | Gt    -> ">"
  | GtE   -> ">="
  | Is    -> "is"
  | IsNot -> "is not"
  | In    -> "in"
  | NotIn -> "not in"


(** Checking various properties of statements *)
let rec contains_continue: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_continue body) 
  | Return _  | Assign _ -> 
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_continue body) || (List.exists contains_continue orelse)
  | With (_, body, _) ->
      List.exists contains_continue body
  | Expr _ | Pass _  | Break _ ->
      false
  | Continue _ ->  
      true

let rec contains_break: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_break body) 
  | Return _  | Assign _ -> 
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_break body) || (List.exists contains_break orelse)
  | With (_, body, _) ->
      List.exists contains_break body
  | Expr _ | Pass _ -> 
      false
  | Break _ ->
      true
  | Continue _ ->  
      false

let rec contains_return: 'a stmt -> bool = function
  | FunctionDef (_, _, body, _, _, _) ->
      (List.exists contains_return body) 
  | Return _  -> 
      true
  | Assign _ -> 
      false
  | For(_, _, body, orelse, _) | While (_, body, orelse, _) | If (_, body, orelse, _) ->
      (List.exists contains_return body) || (List.exists contains_return orelse)
  | With (_, body, _) ->
      List.exists contains_return body
  | Expr _ | Pass _  | Break _ | Continue _ ->  
      false

let contains_middle_return (stmt: 'a stmt): bool =
  match stmt with
  | FunctionDef _ ->
     contains_return stmt
  | Return _ | Assign _  ->
     false
  | For _ | While _ | If _ | With _ ->
     contains_return stmt
  | Expr _  | Pass _ | Break _ | Continue _ ->
     false
