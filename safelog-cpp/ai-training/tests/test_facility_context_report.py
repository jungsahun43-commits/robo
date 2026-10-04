import copy
import unittest
from unittest.mock import patch

from scripts.report_facility_context import (
    AUXILIARY_SHA, CLASSES, CONTEXT_ARCHITECTURE, CONTROL, CONTROL_ARCH, CORE_SHA,
    DOMAINS, EXPECTED, GATE, INITIAL_SHA, LOADER_RANDOMNESS, MATCHED_KEYS, PARENT_SHA,
    REFERENCE, TARGETS, TREATMENT, TREATMENT_ARCH, load_measured_runs,
    optimizer_measurements, resource_measurements, validate_actual_pair, validate_pair,
    validate_technical_verification, research_gate,
)
from tests import test_facility_detail_report as prior_fixtures


class ContextReportTests(unittest.TestCase):
    def protocol(self):
        return {**EXPECTED,'schema':'facility_pool_context_protocol_v1','declared_before_training':True,
                'reference':REFERENCE,'control':CONTROL,'treatment':TREATMENT,'classes':CLASSES,
                'initial_weights_sha256':INITIAL_SHA,'core_spatial_manifest_sha256':CORE_SHA,
                'auxiliary_manifest_sha256':AUXILIARY_SHA,'implementation_parent_sha256':PARENT_SHA,
                'control_architecture':CONTROL_ARCH,'treatment_architecture':TREATMENT_ARCH,
                'target_ranking_weight':0.,'loader_randomness':copy.deepcopy(LOADER_RANDOMNESS),
                'context_architecture':copy.deepcopy(CONTEXT_ARCHITECTURE),'research_candidate_gate':copy.deepcopy(GATE)}

    def pair(self):
        control = {key:'fixed' for key in MATCHED_KEYS}
        control.update(EXPECTED)
        control.update(status='complete',architecture=CONTROL_ARCH,model_variant='control',actual_epochs=6,
            classes=CLASSES,validation_domains=DOMAINS,initial_weights_sha256=INITIAL_SHA,
            spatial_manifest_sha256=CORE_SHA,core_spatial_manifest_sha256=CORE_SHA,
            auxiliary_manifest_sha256=AUXILIARY_SHA,source_parent_training_script_sha256=PARENT_SHA,
            study_protocol_sha256='protocol',study_protocol_path='reports/facility-pool-context-protocol.json',
            loader_randomness=copy.deepcopy(LOADER_RANDOMNESS),
            train_dacl=6225,train_damsegment=1585,train_codebrim=6438,val_dacl=710,val_damsegment=424,
            spatial_manifest_audit={'full_photo_or_patch_rows':14248},
            initial_state_transfer={'shared_state_tensors_equal':True,'shared_state_tensor_count':500,
                                    'new_state_tensor_count':0,'new_output_projection_zero':None},context_architecture=None,
            target_ranking={'weight':0.,'classes':TARGETS,'helper_sha256':'helper','sampling_changed':False,
                            'new_labels_asserted':0,'public_outputs_changed':False},
            expected_sampling={'dacl':{'full':.5,'crop':.2},'damsegment':{'full':.07,'crop':.03},'codebrim':{'full':.2,'crop':0.}},
            expected_label_sampling={label:{'positive':.3,'negative':.6,'unknown':.1} for label in CLASSES})
        treatment = copy.deepcopy(control)
        treatment.update(architecture=TREATMENT_ARCH,model_variant='context',context_architecture=copy.deepcopy(CONTEXT_ARCHITECTURE))
        treatment['initial_state_transfer'].update(new_state_tensor_count=1,new_output_projection_zero=True)
        return control,treatment

    def history(self):
        entry = prior_fixtures.DetailReportTests().history()
        entry['actual_sampling_history'] = entry['actual_sampling_history'][:6]
        return entry

    def entries(self):
        entries = prior_fixtures.DetailReportTests().entries()
        for entry,name in zip(entries,(REFERENCE,CONTROL,TREATMENT)):entry['run']=name
        return entries

    def test_only_declared_zero_signed_gate_pooling_intervention_allowed(self):
        self.assertEqual(validate_pair(*self.pair(),INITIAL_SHA,self.protocol(),'protocol')['seed'],55)
        for mutation in ('topk','new_labels','scene_claim','gate_transform','zero','shared','source','rng','rank','epochs'):
            a,b=self.pair()
            if mutation=='topk':b['context_architecture']['broad_topk']=128
            if mutation=='new_labels':b['context_architecture']['new_photo_labels']=1
            if mutation=='scene_claim':b['context_architecture']['spatial_adjacency_or_scene_semantics_asserted']=True
            if mutation=='gate_transform':b['context_architecture']['gate_transform']='sigmoid'
            if mutation=='zero':b['initial_state_transfer']['new_output_projection_zero']=False
            if mutation=='shared':b['initial_state_transfer']['shared_state_tensor_count']=499
            if mutation=='source':b['context_model_source_sha256']='other'
            if mutation=='rng':b['loader_randomness']['training_worker_seed']=55
            if mutation=='rank':b['target_ranking']['weight']=.25
            if mutation=='epochs':b['actual_epochs']=5
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_pair(a,b,INITIAL_SHA,self.protocol(),'protocol')
        a,b=self.pair();b['detail_model_source_sha256']='another_branch'
        with self.assertRaises(ValueError):validate_pair(a,b,INITIAL_SHA,self.protocol(),'protocol')

    def test_ordered_row_schedule_is_required_for_every_actual_epoch(self):
        self.assertTrue(validate_actual_pair(self.history(),self.history())['actual_ordered_row_index_hashes_identical_each_epoch'])
        for mutation in ('hash','count','unknown','ranking','epochs'):
            a,b=self.history(),self.history();row=b['actual_sampling_history'][3]
            if mutation=='hash':row['row_indices_sha256']='f'*64
            if mutation=='count':row['domain_counts']['dacl']+=1
            if mutation=='unknown':row['full_target_joint_counts']['dacl'].update({'00':2999,'unknown':1})
            if mutation=='ranking':row['ranking_audit']['pair_count']=1
            if mutation=='epochs':b['actual_sampling_history'].append(copy.deepcopy(row))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_actual_pair(a,b)

    def test_reused_error_and_small_case_gates_do_not_grant_deployment(self):
        result=research_gate(*self.entries(),GATE)
        self.assertTrue(result['research_candidate_nominated'])
        self.assertFalse(result['deployment_authorized'])
        a,b,c=self.entries()
        c['small_dacl_polygon_area_below_one_percent'][TARGETS[0]].update(false_negatives=28,fnr=28/93)
        c['small_dacl_polygon_area_below_one_percent'][TARGETS[1]].update(false_negatives=42,fnr=42/105)
        self.assertFalse(research_gate(a,b,c,GATE)['research_candidate_nominated'])

    def test_test_result_is_refused_before_raw_run_or_result_reads(self):
        with patch('scripts.report_facility_context.Path.exists',return_value=True), \
                patch('scripts.report_facility_context.load_run') as loader:
            with self.assertRaises(ValueError):load_measured_runs()
            loader.assert_not_called()

    def test_six_epoch_actual_resources_and_optimizer_skip_totals_are_checked(self):
        history=[{'epoch':epoch,'elapsed_minutes':float(epoch),'peak_cuda_allocated_bytes':1000*epoch,
            'optimizer_step_diagnostics':{'attempted_batches':1781,'actual_optimizer_steps':1781-epoch,'amp_skipped_steps':epoch}}
            for epoch in range(1,7)]
        training={'elapsed_training_minutes':6.5,'peak_cuda_allocated_bytes':6500,
            'optimizer_step_diagnostics':{'attempted_batches':1781*6,'actual_optimizer_steps':1781*6-21,'amp_skipped_steps':21}}
        self.assertEqual(resource_measurements(training,history)['elapsed_training_minutes'],6.5)
        self.assertEqual(optimizer_measurements(training,history)['total']['amp_skipped_steps'],21)
        changed=copy.deepcopy(training);changed['optimizer_step_diagnostics']['actual_optimizer_steps']+=1
        with self.assertRaises(ValueError):optimizer_measurements(changed,history)
        changed=copy.deepcopy(history);changed[3]['elapsed_minutes']=1.
        with self.assertRaises(ValueError):resource_measurements(training,changed)

    def test_technical_zero_gate_single_backbone_and_gradient_are_not_accuracy(self):
        entries=[{'run':REFERENCE,'weights_sha256':'ref'},{'run':CONTROL,'weights_sha256':'a'},{'run':TREATMENT,'weights_sha256':'b'}]
        proof={'schema':'facility_pool_context_technical_verification_v1','status':'passed','study_protocol_sha256':'protocol',
            'app_profile_sha256':'profile','app_profile_unchanged':True,'research_models_deployed':False,
            'accuracy_measured_by_this_check':False,'independent_field_safety_verified':False,
            'paired_sampled_row_indices_equal':True,'new_context_parameter_count':7,
            'experiments':[{'run':entry['run'],'weights_sha256':entry['weights_sha256'],'architecture':arch,
                'model_variant':variant,'study_protocol_sha256':'protocol','public_shape':[1,7],
                'private_spatial_shape':[1,7,80,80],'auxiliary_shape':[1,19],'outputs_finite':True,'public_training_logits_equal':True}
                for entry,arch,variant in zip(entries[1:],(CONTROL_ARCH,TREATMENT_ARCH),('control','context'))],
            'actual_train_gpu_preflight':{'schema':'facility_context_preflight_v1','status':'passed',
                'initial_fp32_outputs_equal':True,'finite_loss_and_gradients':True,'learned_gate_gradient_nonzero':True,
                'backbone_forward_calls':1,'augmented_batch_replay_equal':True},
            'actual_test_verification':{'status':'passed','tests_run':7,'failures':0,'errors':0},
            'pre_training_source_snapshot':{'comparison_started_before_outcomes':True,'pre_training_commit':'frozen',
                'sources':{path:{'equal':True,'git_blob_sha256':'source','executed_file_sha256':'source'} for path in (
                    'scripts/train_facility_context.py','safelog_ai/context_classifier.py','reports/facility-pool-context-protocol.json')}}}
        with patch('scripts.report_facility_context.sha',return_value='source'):
            self.assertFalse(validate_technical_verification(proof,entries,'protocol','profile')['accuracy_measured_by_technical_check'])
            for mutation in ('backbone','gradient','zero','replay','parameter_count','checkpoint','pixels'):
                changed=copy.deepcopy(proof)
                if mutation=='backbone':changed['actual_train_gpu_preflight']['backbone_forward_calls']=2
                if mutation=='gradient':changed['actual_train_gpu_preflight']['learned_gate_gradient_nonzero']=False
                if mutation=='zero':changed['actual_train_gpu_preflight']['initial_fp32_outputs_equal']=False
                if mutation=='replay':changed['actual_train_gpu_preflight']['augmented_batch_replay_equal']=False
                if mutation=='parameter_count':changed['new_context_parameter_count']=8
                if mutation=='checkpoint':changed['experiments'][1]['weights_sha256']='c'
                if mutation=='pixels':changed['experiments'][1]['private_spatial_shape']=[1,7,160,160]
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    validate_technical_verification(changed,entries,'protocol','profile')


if __name__=='__main__':unittest.main()
