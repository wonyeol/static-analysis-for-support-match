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
 ** pyast_sig.ml: abstract syntax tree for Python programs parsed by
 **   Python ast module (i.e., by ast.parse(...) function).
 **
 ** This file is based on ast.ml in ocaml-pythonlib:
 **   https://github.com/m2ym/ocaml-pythonlib/blob/master/src/ast.ml
 ** Note that ast.ml is for ast of Python *2* programs,
 ** while this file  is for ast of Python *3* programs.
 **
 ** The full abstract grammar is available at:
 **   http://docs.python.org/library/ast.html#abstract-grammar
 **)


(* Annotations:
 *   'a denotes the type of annotation.
 * E.g., (lineno, col_offset) : int * int = 'a *)

(* Built-in types *)
type identifier = string
type singleton = bool option

type number =
  | Int     of int
  | Float   of float
  (* | LongInt of int *)
  (* | Imag    of string *)

(* Module Python *)
type 'a modl =
  | Module      of 'a stmt list (* body *) * 'a
  (* | Interactive of 'a stmt list (* body *) * 'a *)
  (* | Expression  of 'a expr      (* body *) * 'a *)
  (* | Suite       of 'a stmt list (* body *) * 'a *)

and 'a stmt =
  (***** FunctionDef: currently unused *****)
  | FunctionDef of identifier (* name *) * 'a arguments (* args *)
        * 'a stmt list (* body *) * 'a expr list (* decorator_list *)
        * 'a expr option (* returns *) * 'a
  (* | AsyncFunctionDef of ... *)

  (* | ClassDef of identifier (* name *) * 'a expr list (* bases *)
       * 'a keyword list (* keywords *)
       * 'a stmt list (* body *) * 'a expr list (* decorator_list *) * 'a *)
  | Return   of 'a expr option (* value *) * 'a

  (* | Delete    of 'a expr list (* targets *) * 'a *)
  | Assign    of 'a expr list (* targets *) * 'a expr (* value *) * 'a
  (* | AugAssign of 'a expr      (* target  *) * operator (* op *)
       * 'a expr (* value *) * 'a *)
  (* | AnnAssign of ... *)

  | For   of 'a expr (* target *) * 'a expr (* iter *)
        * 'a stmt list (* body *) * 'a stmt list (* orelse *) * 'a

  (* | AsyncFor of ... *)

  | While of 'a expr (* test *) * 'a stmt list (* body *)
        * 'a stmt list (* orelse *) * 'a
  | If    of 'a expr (* test *) * 'a stmt list (* body *)
        * 'a stmt list (* orelse *) * 'a
  | With  of 'a withitem list (* items *) * 'a stmt list (* body *) * 'a

  (* | AsyncWith of ... *)
  (* | Raise  of 'a expr option (* exc *) * 'a expr option (* cause *) *)
  (* | Try    of 'a stmt list (* body *) * 'a excepthandler list (* handlers *)
       * 'a stmt list (* orelse *) * 'a stmt list (* finalbody *) * 'a *)
  (* | Assert of 'a expr (* test *) * 'a expr option (* msg *) * 'a *)
  (* | Import     of alias list (* names *) * 'a *)
  (* | ImportFrom of identifier option (* module *) * alias list (* names *)
       * int option (* level *) * 'a *)
  (* | Global   of identifier list (* names *) * 'a *)
  (* | Nonlocal of ... *)

  | Expr     of 'a expr (* value *) * 'a
  | Pass     of 'a
  | Break    of 'a
  | Continue of 'a

and 'a expr =
  | BoolOp   of boolop (* op *) * 'a expr list (* values *) * 'a
  | BinOp    of 'a expr (* left *) * operator (* op *)
        * 'a expr (* right *) * 'a
  | UnaryOp  of unaryop (* op *) * 'a expr (* operand *) * 'a

  (* | Lambda   of 'a arguments (* args *) * 'a expr (* body *) * 'a *)
  (* | IfExp    of 'a expr (* test *) * 'a expr (* body *)
       * 'a expr (* orelse *) * 'a *)
  | Dict     of 'a expr list (* keys *) * 'a expr list (* values *) * 'a
  (* | Set      of 'a expr list (* elts *) *)
  (* | ListComp of 'a expr (* elt *) * 'a comprehension list (* generators *)
       * 'a *)
  (* | SetComp  of ... *)
  (* | DictComp of ... *)
  (* | GeneratorExp of 'a expr (* elt *)
       * 'a comprehension list (* generators *) * 'a *)

  (* | Await of ... *)
  (* | Yield of 'a expr option (* value *) * 'a *)
  (* | YieldFrom of ... *)

  | Compare of 'a expr (* left *) * cmpop list (* ops *)
        * 'a expr list (* comparators *) * 'a
  | Call of 'a expr (* func *) * 'a expr list (* args *)
        * 'a keyword list (* keywords *) * 'a
  | Num of number (* n *) * 'a
  | Str of string (* s *) * 'a

  (* | FormattedValue of 'a expr (* value *) * int option (* conversion *)
       * 'a expr option (* format_spec *) * 'a *)
  (* | JoinedStr of ... *)
  (* | Bytes of ... *)
  | NameConstant of singleton (* value *) * 'a
  (* | Ellipsis of ... *)
  (* | Constant of ... *)

  | Attribute of 'a expr (* value *) * identifier (* attr *)
        * expr_context (* ctx *) * 'a

  | Subscript of 'a expr (* value *) * 'a slice (* slice *)
       * expr_context (* ctx *) * 'a 
  (* | Starred of ... *)

  | Name  of identifier (* id *)     * expr_context (* ctx *) * 'a
  | List  of 'a expr list (* elts *) * expr_context (* ctx *) * 'a
  | Tuple of 'a expr list (* elts *) * expr_context (* ctx *) * 'a

(***** expr_context: currently unused *****)
(* AugLoad and AugStore are not used *)
and expr_context = Load | Store | Del | (* AugLoad | AugStore | *) Param

and 'a slice =
  | Index of 'a expr (* value *)
  | Slice of 'a expr option (* lower *) * 'a expr option (* upper *)
    * 'a expr option (* step *)
  | ExtSlice of 'a slice list (* dims *)

and boolop = And | Or

and operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
               | RShift | BitOr | BitXor | BitAnd | FloorDiv

and unaryop = Invert | Not | UAdd | USub

and cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

(* and 'a comprehension = 'a expr (* target *) * 'a expr (* iter *)
   * 'a expr list (* ifs *) *
                         int (* is_async *) *)

(* and 'a excepthandler = ExceptHandler of 'a expr option (* type *)
   * identifier option (* name *)
   * 'a stmt list (* body *) * 'a *)

(***** arguments: currently unused *****)
and 'a arguments =
    'a expr list (* args *) * 'a arg option (* vararg *)
      * 'a arg list (* kwonlyargs *) * 'a expr list (* kw_defaults *)
      * 'a arg option (* kwargs *) * 'a expr list (* defaults *)

(***** arg: currently unused *****)
and 'a arg = identifier (* arg *) * 'a expr option (* annotation *) * 'a

and 'a keyword = identifier option (* arg *) * 'a expr (* value *)

(* and alias = identifier (* name *) * identifier option (* asname *) *)

and 'a withitem =
    'a expr (* context_expr *) * 'a expr option (* optional_vars *)


(*
(** Annotations *)
(* Lexing.poition = { pos_fname : string, pos_{lnum,bol,cnum} : int } *)
module type Annot =
  sig
    type t
    val of_pos : Lexing.position -> t
    val to_pos : t -> Lexing.position
  end

module Pos : Annot =
  struct
    type t = Lexing.position
    let of_pos pos = pos
    let to_pos pos = pos
  end
 *)
