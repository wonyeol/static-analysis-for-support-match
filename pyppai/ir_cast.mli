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
 ** ir_cast.mli: conversion from Pyast_sig to Ir_sig *)
open Ir_sig
module Pya = Pyast_sig

(** Conversion of basic types *)
val dist_kind_list: string list
val to_dist_kind: string -> dist_kind

(** Conversion of main language components *)
val modl_to_prog: 'a Pya.modl -> prog
