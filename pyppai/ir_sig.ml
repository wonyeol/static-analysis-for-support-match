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
 ** ir_sig.ml: signature of the intermediate representation syntax treew
 **            obtained by elaboration from pyast_sig AST *)

(* Exception for must error *)
exception Must_error of string

(* Identifiers *)
type idtf = string

(* Values *)
type number =
  | Int   of int
  | Float of float

(* Operators *)
type uop =
  | Not
  | SampledStr
  | SampledStrFmt
type bop =
  | And | Or
  | Add | Sub | Mult | Div | Pow
type cop =
  | Eq | NotEq | Lt | LtE | Gt | GtE | Is | NotIs

(* Expressions *)
type expr =
  | Nil
  | True | False
  | Num  of number (* n *)
  | Name of idtf (* id *)

  | UOp  of uop (* op *) * expr (* v *)
  | BOp  of bop (* op *) * expr (* el *) * expr (* er *)
  | Comp of cop (* op *) * expr (* el *) * expr (* er *)
  | List of expr list (* es *)
  | Dict of expr list (* ks *) * expr list (* vs *)

  | Str    of string (* s *)
  | StrFmt of string (* s *) * expr list (* vs *)

(* Distributions *)
type dist_kind =
  (* continuous *)
  | Normal      (* supp = (-\infty,\infty) *)
  | Exponential (* supp = [0,\infty) *)
  | Gamma       (* supp = (0,\infty) *)
  | Beta        (* supp = (0,1) *)
  | Uniform of (float (* lower bound *) * float (* upper bound *)) option
                (* supp = [l,u] *)
  | Dirichlet of int option (* N>=2: the number of cases *)
                (* supp = (N-1) simplex \subset \R^N *)
  (* discrete *)
  | Poisson     (* supp = {0,1,...} *)
  | Categorical of int option (* N>=2: the number of cases *)
                (* supp = {0,1,...,N-1} *)
  | Bernoulli   (* supp = {0,1} *)
  | OneHotCategorical of int option (* N>=2: the number of cases *)
                (* supp = {0,1}^N *)
  | Delta       (* supp = {v} *)
  (* subsample *)
  | Subsample of int option (* total size *) * int option (* subsambled size *)

(* Operators for changing distribution objects, in particular,
 * their dimensions. *)
type dist_trans =
  | ExpandBy of int option list
  | ToEvent of int option

type dist = dist_kind * dist_trans list (* * broadcast_info *)

(* Keyword *)
type keyword =
  idtf option (* arg *) * expr (* value *)

(* Analysis commands *)
type acmd  =
  | Assert   of expr (* cond *)
  | Assume   of expr (* cond *)
  | Assn     of idtf (* trgt *) * expr (* v *)
  | AssnCall of idtf (* trgt *) * expr (* func *) * expr list (* args *)
        * keyword list (* kwargs *)
  | Sample   of idtf (* trgt *) * expr (* name *) * dist (* d *)
        * expr list (* args *) 
        * expr option (* obsv if some *)

(* Components of withitem statements *)
type withitem = expr * expr option

(* program commands *)
type stmt =
  | Atomic of acmd (* ac *)
  | If     of expr (* cond *) * block (* body *) * block (* orelse *)
  | For    of expr (* i *) * expr (* range *) * block (* body *)
  | While  of expr (* cond *) * block (* body *)
  | With   of withitem list (* items *) * block (* body *)
  | Break
  | Continue

and block = stmt list

type prog = stmt list
