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
 ** pyast_cast.mli: conversion from Py.Object.t to Pyast_sig *)
open Pyast_sig
       
(** Conversion of main language component *)
val pyobj_to_modl: Py.Object.t -> 'a option modl
