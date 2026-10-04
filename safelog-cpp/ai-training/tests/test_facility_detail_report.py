import copy
import unittest
from unittest.mock import patch

from scripts.report_facility_detail import (
    AUXILIARY_SHA, CLASSES, CONTROL, CONTROL_ARCH, CORE_SHA, COUNTS, DETAIL_ARCHITECTURE,
    DOMAINS, EXPECTED, GATE, INITIAL_SHA, LOADER_RANDOMNESS, MATCHED_KEYS, PARENT_SHA,
    REFERENCE, TARGETS, TREATMENT, TREATMENT_ARCH, detail_research_gate,
    load_measured_runs, optimizer_measurements, resource_measurements,
    validate_actual_pair, validate_pair, validate_technical_verification,
)
from tests import test_facility_discrimination_report as prior_fixtures


class DetailReportTests(unittest.TestCase):
    def protocol(self):
        return {**EXPECTED, 'schema':'facility_detail_architecture_protocol_v1', 'declared_before_training':True,
                'reference':REFERENCE, 'control':CONTROL, 'treatment':TREATMENT, 'classes':CLASSES,
                'initial_weights_sha256':INITIAL_SHA, 'core_spatial_manifest_sha256':CORE_SHA,
                'auxiliary_manifest_sha256':AUXILIARY_SHA, 'implementation_parent_sha256':PARENT_SHA,
                'control_architecture':CONTROL_ARCH, 'treatment_architecture':TREATMENT_ARCH,
                'target_ranking_weight':0., 'loader_randomness':LOADER_RANDOMNESS,
                'detail_architecture':DETAIL_ARCHITECTURE, 'research_candidate_gate':GATE}

    def pair(self):
        control = {key:'fixed' for key in MATCHED_KEYS}
        control.update(EXPECTED)
        control.update(status='complete', architecture=CONTROL_ARCH, model_variant='control',
                       actual_epochs=8, classes=CLASSES, validation_domains=DOMAINS,
                       initial_weights_sha256=INITIAL_SHA, spatial_manifest_sha256=CORE_SHA,
                       core_spatial_manifest_sha256=CORE_SHA, auxiliary_manifest_sha256=AUXILIARY_SHA,
                       source_parent_training_script_sha256=PARENT_SHA,
                       study_protocol_sha256='protocol', study_protocol_path='reports/facility-detail-architecture-protocol.json',
                       loader_randomness=copy.deepcopy(LOADER_RANDOMNESS),
                       train_dacl=6225, train_damsegment=1585, train_codebrim=6438, val_dacl=710, val_damsegment=424,
                       spatial_manifest_audit={'full_photo_or_patch_rows':14248},
                       initial_state_transfer={'shared_state_tensors_equal':True,'shared_state_tensor_count':500,
                                               'new_state_tensor_count':0,'new_output_projection_zero':None},
                       detail_architecture=None,
                       target_ranking={'weight':0.,'classes':TARGETS,'helper_sha256':'helper',
                                       'sampling_changed':False,'new_labels_asserted':0,'public_outputs_changed':False},
                       expected_sampling={'dacl':{'full':.5,'crop':.2},'damsegment':{'full':.07,'crop':.03},
                                          'codebrim':{'full':.2,'crop':0.}},
                       expected_label_sampling={label:{'positive':.3,'negative':.6,'unknown':.1} for label in CLASSES})
        treatment = copy.deepcopy(control)
        treatment.update(architecture=TREATMENT_ARCH, model_variant='detail', detail_architecture=copy.deepcopy(DETAIL_ARCHITECTURE))
        treatment['initial_state_transfer'].update(new_state_tensor_count=8,new_output_projection_zero=True)
        return control,treatment

    def history(self):
        rows = prior_fixtures.DiscriminationReportTests().history(False)['actual_sampling_history']
        rows += [copy.deepcopy(rows[0]),copy.deepcopy(rows[0])]
        for epoch,row in enumerate(rows,1):
            row.update(epoch=epoch,row_indices_sha256=f'{epoch:064x}')
        return {'actual_sampling_history':rows}

    def entries(self):
        entries = prior_fixtures.DiscriminationReportTests().entries()
        for entry,name,misses in zip(entries,(REFERENCE,CONTROL,TREATMENT),((29,42),(31,42),(27,40))):
            entry['run']=name
            entry['small_dacl_polygon_area_below_one_percent'] = {
                label:{'positive_photos':n,'false_negatives':fn,'fnr':fn/n}
                for label,n,fn in zip(TARGETS,(93,105),misses)}
        return entries

    def test_only_declared_architecture_difference_and_zero_base_transfer_allowed(self):
        result = validate_pair(*self.pair(),INITIAL_SHA,self.protocol(),'protocol')
        self.assertEqual(result['loader_randomness'],LOADER_RANDOMNESS)
        for mutation in ('architecture','supervision','rng','manifest','zero','shared','source','rank','protocol','epochs'):
            a,b=self.pair()
            if mutation=='architecture':b['architecture']='larger_undeclared'
            if mutation=='supervision':b['detail_architecture']['finer_pixel_gold_asserted']=True
            if mutation=='rng':b['loader_randomness']['training_worker_seed']=54
            if mutation=='manifest':b['core_spatial_manifest_sha256']='different'
            if mutation=='zero':b['initial_state_transfer']['new_output_projection_zero']=False
            if mutation=='shared':b['initial_state_transfer']['shared_state_tensor_count']=499
            if mutation=='source':b['detail_model_source_sha256']='other'
            if mutation=='rank':b['target_ranking']['weight']=.25
            if mutation=='protocol':b['study_protocol_sha256']='other'
            if mutation=='epochs':b['actual_epochs']=7
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_pair(a,b,INITIAL_SHA,self.protocol(),'protocol')

    def test_small_area_gate_requires_two_case_improvement_against_both_comparators(self):
        entries=self.entries()
        result=detail_research_gate(*entries,GATE)
        self.assertTrue(result['research_candidate_nominated'])
        self.assertFalse(result['deployment_authorized'])
        self.assertFalse(result['small_area_photo_label_case_totals']['treatment']['unique_photo_count_established'])
        for mutation in ('one_case','one_class_regression','wrong_support','wrong_numeric_rate','weaken_gate'):
            a,b,c=self.entries();declared=copy.deepcopy(GATE)
            if mutation=='one_case':
                c['small_dacl_polygon_area_below_one_percent'][TARGETS[0]].update(false_negatives=28,fnr=28/93)
                c['small_dacl_polygon_area_below_one_percent'][TARGETS[1]].update(false_negatives=42,fnr=42/105)
            if mutation=='one_class_regression':
                c['small_dacl_polygon_area_below_one_percent'][TARGETS[0]].update(false_negatives=31,fnr=31/93)
                c['small_dacl_polygon_area_below_one_percent'][TARGETS[1]].update(false_negatives=36,fnr=36/105)
            if mutation=='wrong_support':c['small_dacl_polygon_area_below_one_percent'][TARGETS[0]]['positive_photos']=94
            if mutation=='wrong_numeric_rate':c['small_dacl_polygon_area_below_one_percent'][TARGETS[0]]['fnr']=.01
            if mutation=='weaken_gate':declared['minimum_small_combined_fn_reduction_vs_reference_and_control']=1
            with self.subTest(mutation=mutation):
                if mutation in ('wrong_support','wrong_numeric_rate','weaken_gate'):
                    with self.assertRaises(ValueError):detail_research_gate(a,b,c,declared)
                else:self.assertFalse(detail_research_gate(a,b,c,declared)['research_candidate_nominated'])

    def test_sampling_counts_are_insufficient_without_identical_ordered_row_hashes(self):
        self.assertTrue(validate_actual_pair(self.history(),self.history())['actual_ordered_row_index_hashes_identical_each_epoch'])
        for mutation in ('index_order','missing_hash','unknown','budget','ranking','epoch_count'):
            a,b=self.history(),self.history();row=b['actual_sampling_history'][3]
            if mutation=='index_order':row['row_indices_sha256']='f'*64
            if mutation=='missing_hash':row.pop('row_indices_sha256')
            if mutation=='unknown':row['full_target_joint_counts']['dacl'].update({'00':2999,'unknown':1})
            if mutation=='budget':row['domain_counts']['dacl']+=1
            if mutation=='ranking':row['ranking_audit']['pair_count']=10
            if mutation=='epoch_count':b['actual_sampling_history'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_actual_pair(a,b)

    def test_refuses_test_artifact_before_any_run_loading_or_result_read(self):
        with patch('scripts.report_facility_detail.Path.exists',return_value=True), \
                patch('scripts.report_facility_detail.load_run') as loader:
            with self.assertRaises(ValueError):load_measured_runs()
            loader.assert_not_called()

    def test_actual_resources_are_cumulative_observations_not_equal_cost_assumption(self):
        history=[{'elapsed_minutes':float(epoch),'peak_cuda_allocated_bytes':1000*epoch} for epoch in range(1,9)]
        training={'elapsed_training_minutes':8.5,'peak_cuda_allocated_bytes':8500}
        result=resource_measurements(training,history)
        self.assertEqual(result['elapsed_training_minutes'],8.5)
        history[4]['elapsed_minutes']=1.
        with self.assertRaises(ValueError):resource_measurements(training,history)

    def test_actual_optimizer_update_counts_may_differ_but_totals_must_be_consistent(self):
        history=[{'epoch':epoch,'optimizer_step_diagnostics':{'attempted_batches':1781,
                  'actual_optimizer_steps':1781-epoch,'amp_skipped_steps':epoch}} for epoch in range(1,9)]
        totals={'attempted_batches':1781*8,'actual_optimizer_steps':1781*8-36,'amp_skipped_steps':36}
        result=optimizer_measurements({'optimizer_step_diagnostics':totals},history)
        self.assertEqual(result['total']['amp_skipped_steps'],36)
        totals['actual_optimizer_steps']+=1
        with self.assertRaises(ValueError):optimizer_measurements({'optimizer_step_diagnostics':totals},history)

    def test_technical_transfer_and_augmentation_proof_is_not_accuracy(self):
        entries=[{'run':REFERENCE,'weights_sha256':'ref'}, {'run':CONTROL,'weights_sha256':'a'},
                 {'run':TREATMENT,'weights_sha256':'b'}]
        proof={'status':'passed','study_protocol_sha256':'protocol','app_profile_sha256':'profile',
               'app_profile_unchanged':True,'research_models_deployed':False,'paired_sampled_row_indices_equal':True,
               'experiments':[{'run':entry['run'],'weights_sha256':entry['weights_sha256'],
                    'architecture':arch,'model_variant':variant,'study_protocol_sha256':'protocol',
                    'public_shape':[1,7],'private_spatial_shape':[1,7,80,80],'auxiliary_shape':[1,19],
                    'outputs_finite':True,'public_training_logits_equal':True}
                    for entry,arch,variant in zip(entries[1:],(CONTROL_ARCH,TREATMENT_ARCH),('control','detail'))],
               'actual_train_gpu_preflight':{'status':'passed','initial_fp32_outputs_equal':True,
                    'finite_loss_and_gradients':True,'detail_upstream_gradient_after_projection_step':True,
                    'augmented_batch_replay_equal':True},
               'actual_test_verification':{'status':'passed','tests_run':9,'failures':0,'errors':0},
               'pre_training_source_snapshot':{'comparison_started_before_outcomes':True,'pre_training_commit':'frozen',
                    'sources':{path:{'equal':True,'git_blob_sha256':'source','executed_file_sha256':'source'} for path in (
                    'scripts/train_facility_detail.py','safelog_ai/detail_classifier.py','reports/facility-detail-architecture-protocol.json')}}}
        with patch('scripts.report_facility_detail.sha',return_value='source'):
            result=validate_technical_verification(proof,entries,'protocol','profile')
            self.assertFalse(result['accuracy_measured_by_technical_check'])
            for mutation in ('new_pixels','augmentation','zero_projection','checkpoint','source_snapshot'):
                changed=copy.deepcopy(proof)
                if mutation=='new_pixels':changed['experiments'][1]['private_spatial_shape']=[1,7,160,160]
                if mutation=='augmentation':changed['actual_train_gpu_preflight']['augmented_batch_replay_equal']=False
                if mutation=='zero_projection':changed['actual_train_gpu_preflight']['initial_fp32_outputs_equal']=False
                if mutation=='checkpoint':changed['experiments'][1]['weights_sha256']='c'
                if mutation=='source_snapshot':changed['pre_training_source_snapshot']['comparison_started_before_outcomes']=False
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    validate_technical_verification(changed,entries,'protocol','profile')


if __name__=='__main__':unittest.main()
