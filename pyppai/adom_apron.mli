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
 ** adom_apron.ml: Numerical abstract domain based on Apron *)
open Adom_sig
open Apron

(** Available managers *)
module PA_box: APRON_MGR
module PA_oct: APRON_MGR
module PA_pol: APRON_MGR

(** Apron domain constructor *)
module Make: APRON_MGR -> ABST_DOMAIN_NB_D
