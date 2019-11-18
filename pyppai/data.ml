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
 ** data.ml: data programs to be used for batch testing (POPL'20) *)

open Main

type use_zone = Zone | NoZone
type test_descr =
    { td_name:   string ;
      td_model:  string ;
      td_guide:  string ;
      td_zone:   use_zone ;
      td_result: test_oracle_r }

let compute_options td =
  let ao_zone = td.td_zone = Zone in
  let model =
    { aopts_default with
      ao_do_num   = Some AD_oct;
      ao_input    = AI_pyfile td.td_model;
      ao_sim_assm = true;
      ao_zone     = ao_zone } in
  let guide =
    { aopts_default with
      ao_do_num   = Some AD_oct;
      ao_input    = AI_pyfile td.td_guide;
      ao_sim_assm = true;
      ao_zone     = ao_zone } in
  (td.td_name, model, guide, td.td_result)

let pyro_test_suite =
  (* 
   * 0-plate = no plates
   * 1-iplate = single for-plate
   * n-iplate = nested for-plates
   * 1-wplate = single with-plate
   * n-wplate = nested with-plates
   * s-wplate = non-nested with-plates
   * n-iwplate = nested for-plate & with-plate 
   * ssz = subsample_size
   *)
  [ { td_name   = "suite, 0-  plate, nonempty_model_empty_guide_ok";
      td_model  = "./test/pyro_test_suite/model0.py";
      td_guide  = "./test/pyro_test_suite/guide0.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 0-  plate, empty_model_empty_guide_ok";
      td_model  = "./test/pyro_test_suite/model1.py";
      td_guide  = "./test/pyro_test_suite/guide1.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 0-  plate, variable_clash_in_model_error";
      td_model  = "./test/pyro_test_suite/model2.py";
      td_guide  = "./test/pyro_test_suite/guide2.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, 0-  plate, model_guide_dim_mismatch_error ";
      td_model  = "./test/pyro_test_suite/model3.py";
      td_guide  = "./test/pyro_test_suite/guide3.py";
      td_zone   = NoZone ;
      td_result = TOR_fail } ;
    { td_name   = "suite, 0-  plate, model_guide_shape_mismatch_error ";
      td_model  = "./test/pyro_test_suite/model4.py";
      td_guide  = "./test/pyro_test_suite/guide4.py";
      td_zone   = NoZone ;
      td_result = TOR_fail } ;
    { td_name   = "suite, 0-  plate, variable_clash_in_guide_error";
      td_model  = "./test/pyro_test_suite/model5.py";
      td_guide  = "./test/pyro_test_suite/guide5.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, 0-  plate, enum_discrete_misuse_warning";
      td_model  = "./test/pyro_test_suite/model32.py";
      td_guide  = "./test/pyro_test_suite/guide32.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 0-  plate, mean_field_ok";
      td_model  = "./test/pyro_test_suite/model37.py";
      td_guide  = "./test/pyro_test_suite/guide37.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 0-  plate, mean_field_warn";
      td_model  = "./test/pyro_test_suite/model38.py";
      td_guide  = "./test/pyro_test_suite/guide38.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- iplate, iplate_in_model_not_guide_ok (ssz=None)";
      td_model  = "./test/pyro_test_suite/model16.py";
      td_guide  = "./test/pyro_test_suite/guide16.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- iplate, iplate_in_model_not_guide_ok (ssz=5)";
      td_model  = "./test/pyro_test_suite/model17.py";
      td_guide  = "./test/pyro_test_suite/guide17.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- iplate, iplate_in_guide_not_model_error (ssz=None)";
      td_model  = "./test/pyro_test_suite/model18.py";
      td_guide  = "./test/pyro_test_suite/guide18.py";
      td_zone   = NoZone ;
      td_result = TOR_fail } ;
    { td_name   = "suite, 1- iplate, iplate_in_guide_not_model_error (ssz=5)";
      td_model  = "./test/pyro_test_suite/model19.py";
      td_guide  = "./test/pyro_test_suite/guide19.py";
      td_zone   = NoZone ;
      td_result = TOR_fail } ;
    { td_name   = "suite, 1- iplate, iplate_ok (ssz=None)";
      td_model  = "./test/pyro_test_suite/model6.py";
      td_guide  = "./test/pyro_test_suite/guide6.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- iplate, iplate_ok (ssz=2)";
      td_model  = "./test/pyro_test_suite/model7.py";
      td_guide  = "./test/pyro_test_suite/guide7.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- iplate, iplate_variable_clash_error";
      td_model  = "./test/pyro_test_suite/model8.py";
      td_guide  = "./test/pyro_test_suite/guide8.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, n- iplate, iplate_iplate_ok (ssz=None)";
      td_model  = "./test/pyro_test_suite/model12.py";
      td_guide  = "./test/pyro_test_suite/guide12.py";
      td_zone   = Zone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- iplate, iplate_iplate_ok (ssz=2)";
      td_model  = "./test/pyro_test_suite/model13.py";
      td_guide  = "./test/pyro_test_suite/guide13.py";
      td_zone   = Zone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- iplate, iplate_iplate_swap_ok (ssz=None)";
      td_model  = "./test/pyro_test_suite/model14.py";
      td_guide  = "./test/pyro_test_suite/guide14.py";
      td_zone   = Zone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- iplate, iplate_iplate_swap_ok (ssz=2)";
      td_model  = "./test/pyro_test_suite/model15.py";
      td_guide  = "./test/pyro_test_suite/guide15.py";
      td_zone   = Zone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- wplate, plate_ok (ssz=None)";
      td_model  = "./test/pyro_test_suite/model9.py";
      td_guide  = "./test/pyro_test_suite/guide9.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- wplate, plate_ok (ssz=5)";
      td_model  = "./test/pyro_test_suite/model10.py";
      td_guide  = "./test/pyro_test_suite/guide10.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- wplate, plate_no_size_ok";
      td_model  = "./test/pyro_test_suite/model11.py";
      td_guide  = "./test/pyro_test_suite/guide11.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, 1- wplate, plate_broadcast_error";
      td_model  = "./test/pyro_test_suite/model20.py";
      td_guide  = "./test/pyro_test_suite/guide20.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, 1- wplate, plate_wrong_size_error";
      td_model  = "./test/pyro_test_suite/model31.py";
      td_guide  = "./test/pyro_test_suite/guide31.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, n- wplate, nested_plate_plate_ok";
      td_model  = "./test/pyro_test_suite/model23.py";
      td_guide  = "./test/pyro_test_suite/guide23.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- wplate, nested_plate_plate_dim_error_1";
      td_model  = "./test/pyro_test_suite/model25.py";
      td_guide  = "./test/pyro_test_suite/guide25.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- wplate, nested_plate_plate_dim_error_2";
      td_model  = "./test/pyro_test_suite/model26.py";
      td_guide  = "./test/pyro_test_suite/guide26.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, n- wplate, nested_plate_plate_dim_error_3";
      td_model  = "./test/pyro_test_suite/model27.py";
      td_guide  = "./test/pyro_test_suite/guide27.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- wplate, nested_plate_plate_dim_error_4";
      td_model  = "./test/pyro_test_suite/model28.py";
      td_guide  = "./test/pyro_test_suite/guide28.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, n- wplate, plate_shape_broadcasting";
      td_model  = "./test/pyro_test_suite/model33.py";
      td_guide  = "./test/pyro_test_suite/guide33.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- wplate, dim_allocation_ok";
      td_model  = "./test/pyro_test_suite/model34.py";
      td_guide  = "./test/pyro_test_suite/guide34.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n- wplate, dim_allocation_error";
      td_model  = "./test/pyro_test_suite/model35.py";
      td_guide  = "./test/pyro_test_suite/guide35.py";
      td_zone   = NoZone ;
      td_result = TOR_error } ;
    { td_name   = "suite, n- wplate, vectorized_num_particles";
      td_model  = "./test/pyro_test_suite/model36.py";
      td_guide  = "./test/pyro_test_suite/guide36.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, s- wplate, plate_reuse_ok";
      td_model  = "./test/pyro_test_suite/model24.py";
      td_guide  = "./test/pyro_test_suite/guide24.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, s- wplate, nonnested_plate_plate_ok";
      td_model  = "./test/pyro_test_suite/model29.py";
      td_guide  = "./test/pyro_test_suite/guide29.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n-iwplate, plate_iplate_ok";
      td_model  = "./test/pyro_test_suite/model21.py";
      td_guide  = "./test/pyro_test_suite/guide21.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n-iwplate, iplate_plate_ok";
      td_model  = "./test/pyro_test_suite/model22.py";
      td_guide  = "./test/pyro_test_suite/guide22.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
    { td_name   = "suite, n-iwplate, three_indep_plate_at_different_depths_ok";
      td_model  = "./test/pyro_test_suite/model30.py";
      td_guide  = "./test/pyro_test_suite/guide30.py";
      td_zone   = NoZone ;
      td_result = TOR_succeed } ;
  ]

let pyro_examples =
  [ { td_name   = "BR example, original";
      td_model  = "./test/pyro_example/br_model0.py";
      td_guide  = "./test/pyro_example/br_guide0.py";
      td_zone   = NoZone;
      td_result = TOR_fail };
    { td_name   = "BR example, edited";
      td_model  = "./test/pyro_example/br_model1.py";
      td_guide  = "./test/pyro_example/br_guide1.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "CSIS example, opposite order";
      td_model  = "./test/pyro_example/csis_guide.py";
      td_guide  = "./test/pyro_example/csis_model.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "LDA example, original";
      td_model  = "./test/pyro_example/lda_model.py";
      td_guide  = "./test/pyro_example/lda_guide0.py";
      td_zone   = NoZone;
      td_result = TOR_fail };
    { td_name   = "LDA example, edited 1";
      td_model  = "./test/pyro_example/lda_model.py";
      td_guide  = "./test/pyro_example/lda_guide1.py";
      td_zone   = NoZone;
      td_result = TOR_fail };
    { td_name   = "LDA example, edited 2";
      td_model  = "./test/pyro_example/lda_model.py";
      td_guide  = "./test/pyro_example/lda_guide2.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "VAE example";
      td_model  = "./test/pyro_example/vae_model.py";
      td_guide  = "./test/pyro_example/vae_guide.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "SGDEF example";
      td_model  = "./test/pyro_example/sgdef_model.py";
      td_guide  = "./test/pyro_example/sgdef_guide.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "DMM example";
      td_model  = "./test/pyro_example/dmm_model.py";
      td_guide  = "./test/pyro_example/dmm_guide.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "SSVAE example, 0";
      td_model  = "./test/pyro_example/ssvae_model0.py";
      td_guide  = "./test/pyro_example/ssvae_guide0.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "SSVAE example, 1";
      td_model  = "./test/pyro_example/ssvae_model1.py";
      td_guide  = "./test/pyro_example/ssvae_guide1.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
    { td_name   = "AIR example";
      td_model  = "./test/pyro_example/air_model.py";
      td_guide  = "./test/pyro_example/air_guide.py";
      td_zone   = NoZone;
      td_result = TOR_succeed };
 ]


let suite    = List.map compute_options pyro_test_suite
let examples = List.map compute_options pyro_examples
