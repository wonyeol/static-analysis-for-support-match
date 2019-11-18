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
 ** pyast_util.mli: utilities on Python AST *)
open Pyast_sig

(** Extraction of names *)
val name_of_number:       number       -> string
val name_of_modl:         'a modl      -> string
val name_of_stmt:         'a stmt      -> string
val name_of_expr:         'a expr      -> string
val name_of_expr_context: expr_context -> string
val name_of_boolop:       boolop       -> string
val name_of_operator:     operator     -> string
val name_of_unaryop:      unaryop      -> string
val name_of_cmpop:        cmpop        -> string

(** Conversion to strings *)
val string_of_number:     number       -> string
val string_of_boolop:     boolop       -> string
val string_of_operator:   operator     -> string
val string_of_unaryop:    unaryop      -> string
val string_of_cmpop:      cmpop        -> string

(** Checking various properties of statements *)
val contains_continue: 'a stmt -> bool
val contains_break: 'a stmt -> bool
val contains_return: 'a stmt -> bool
val contains_middle_return: 'a stmt -> bool
