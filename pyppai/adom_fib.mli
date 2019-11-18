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
 ** adom_fib.ml: prototype implementation of a fibered domain functor *)
open Adom_sig

type maymust = Must | May
val dim_sample_low_make: maymust -> string -> string 
val dim_sample_high_make: maymust -> string -> string 
val dim_tens_size_make: string -> string

module Make: ABST_DOMAIN_ZONE_NB -> ABST_DOMAIN_NB_D -> ABST_DOMAIN_NB
