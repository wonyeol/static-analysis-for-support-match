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
 ** irty_sig.ml: signature of the type of an expr *)

open Lib
open Ir_sig

(**********)
(* exp_ty *)
(**********)
(* Base numeric types (also used as is in adom_fib) 
 *
 * - Implicit subtyping relationship NT_int <: NT_real *)
type num_ty =
  | NT_int
  | NT_real

(* Types for tensor sizes 
 *
 * When (None) is a tensor size, the dimension of the tensor is unknown.
 * When (Some l) is a tensor size,
 *   (a) (List.length l) represents the number of dimensions of a tensor;
 *   (b) if (List.nth l n) = Some(m), the n-th dimension of the tensor is m; and
 *   (c) if (List.nth l n) = None, the n-th dimension of the tensor is unknown.
 *)
type tensor_size_ty = (int option list) option

(* Type for plate objects 
 *
 * - The first part of a pair represents the integer arguments used to create a plate.
 *   For instance, [], [total-size], [total-size;subsample-size].
 * - The second part represents some part (for now, only 'dim' is considered)
 *   of the keyword arguments. For instance, [dim:-1].
 *)
type plate_ty = ((int option list) * (int option SM.t)) option

(* Type for range objects 
 *
 * - When `None' is a range_ty, the arguments of a range object is unknown.
 * - When `Some l' is a range_ty, the arguments of a range object is `l'.
 *)
type range_ty = (int option list) option

(* Type for pyro distribution objects 
 *
 * - The first part of a pair represents the constructor used to create a distribution.
 * - The second part represents the tensor size of the arguments used to create a distribution.
 *)
type distr_ty = (dist_kind * tensor_size_ty list) option

(* type for functions *)
type fun_ty = 
  (* function on tensors that transforms the size of the input tensor. 
   * - FT_tens_resize([i1;i2;...;in], [j1;j2;...;jm], [k1;...;kl]) maps
   *   tensors of size [l1;...;lk; i1;...;in] to
   *   [k1;...;kl; l1;...;lk; j1;...;jm] *)
  | FT_tens_resize of (int option list) * (int option list) * (int option list)
  (* any functions *)
  | FT_top

(* Types for expressions. 
 *
 * 1. ET_bool: Type for booleans.
 * 2. ET_nil: Type for nil.
 * 3. ET_num(nt): Type for numbers.
 * 4. ET_plate(pt): Type for plates.
 * 5. ET_fun(ft): Type for functions.
 * 6. ET_tensor(ts): Type for tensors.
 * 7. ET_unknown: Type for anything. It is used when we don't know the type of 
 *      a given expression. 
 *)
type exp_ty = 
  | ET_nil 
  | ET_bool 
  | ET_num of num_ty 
  | ET_tensor of tensor_size_ty
  | ET_plate of plate_ty
  | ET_range of range_ty
  | ET_distr of distr_ty
  | ET_fun of fun_ty 
  | ET_unknown

(********)
(* vtyp *)
(********)
(* Information about the types of variables *)
(* Meaning of tensor dimensions:
 *    None                      :  top
 *    [None, Some 3, Some 4]    :  ? x 3 x 4 tensor *)
type vtyp =
  | Vt_nil
  | Vt_num of num_ty
  | Vt_tens of tensor_size_ty
  | Vt_plate of plate_ty
  | Vt_range of range_ty
  | Vt_distr of distr_ty
  | Vt_fun of fun_ty
                 
(* hy: One thing that we discussed is that Vt_.. should
 * be merged to ET_.. If denotable values by expressions
 * and storable values in variables are different, such
 * distinction makes sense. If not (which is our current setting), 
 * this distinction may or may not be a good choice. It encodes 
 * the fact that the analysis does not track the type information 
 * of boolean variables and it expresses the lack of type information 
 * in some other means, such as the partiality of a map. *)

(******************)
(* broadcast_info *)
(******************)
(* Information about broadcasting imposed by pyro context
 * statements such as "with pyro.plate".
 *
 * - The first component of a list is for the size of the
 *     broadcasted dimension.
 * - The second component is for a particular dimension
 *     where the broadcasting is applied.
 * - The order is important. There is a default decreasing
 *     counting from -1 and the value of this counting is
 *     used to handle a pair with None as its second
 *     component. *)
type broadcast_info =
  (int * int option) list

(****************)  
(* dist_size_ty *)  
(****************)  
(* Type for distribution objects.
 *
 * - It consists of two parts, batch_size_ty and event_size_ty.
 * - In pyro, there is one more part called sample dimension.
 *     We do not consider it at this point.
 *)
type batch_size_ty = (int option list) 
type event_size_ty = (int option list) 
type dist_size_ty = batch_size_ty * event_size_ty
