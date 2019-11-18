(** pyppai: basic abstract interpreter for python probabilistic programs
 **
 ** Authors:
 **  Wonyeol Lee, KAIST
 **  Xavier Rival, INRIA Paris
 **  Hongseok Yang, KAIST
 **  Hangyeol Yu, KAIST
 **
 ** Copyright (c) 2019 KAIST and INRIA Paris
 ** 
 ** pyobj_util.mli: util for Py.Object.t *)

(** construct ast *)
val get_ast: string -> Py.Object.t

(** extract info *)
val get_attr: Py.Object.t -> string -> Py.Object.t
val get_classname: Py.Object.t -> string

(** print *)
val to_string: Py.Object.t -> string
val print: Py.Object.t -> unit
