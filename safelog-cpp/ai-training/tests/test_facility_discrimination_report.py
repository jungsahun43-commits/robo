import copy
import unittest
from unittest.mock import patch

from scripts.report_facility_discrimination import (
    CLASSES, COUNTS, DOMAINS, TARGETS, KNOWN_INDICES, EXPECTED, MATCHED_KEYS,
    GATE, INITIAL_SHA, REFERENCE, CONTROL, TREATMENT, average_precision,
    descriptive_interval, research_gate, validate_pair, validate_actual_pair,
    verify_validation_cache, validate_train_audit,
    validate_technical_verification,
)


class DiscriminationReportTests(unittest.TestCase):
    def protocol(self):
        return {**EXPECTED, 'schema': 'facility_target_discrimination_protocol_v1',
                'declared_before_training': True, 'reference': REFERENCE, 'control': CONTROL,
                'treatment': TREATMENT, 'initial_weights_sha256': INITIAL_SHA, 'classes': CLASSES,
                'control_ranking_weight': 0., 'treatment_ranking_weight': .25,
                'research_candidate_gate': GATE}

    def pair(self):
        control = {key: 'fixed' for key in MATCHED_KEYS}
        control.update(EXPECTED)
        control.update(status='complete', architecture='lraspp_mobilenet_facility_auxiliary_v1',
                       classes=CLASSES, actual_epochs=6, validation_domains=DOMAINS,
                       initial_weights_sha256=INITIAL_SHA, train_dacl=6225, train_damsegment=1585,
                       train_codebrim=6438, val_dacl=710, val_damsegment=424,
                       spatial_manifest_sha256='core', core_spatial_manifest_sha256='core',
                       spatial_manifest_audit={'full_photo_or_patch_rows':14248},
                       study_protocol_sha256='protocol',
                       study_protocol_path='reports/facility-target-discrimination-protocol.json',
                       expected_sampling={'dacl':{'full':.5,'crop':.2}, 'damsegment':{'full':.07,'crop':.03},
                                          'codebrim':{'full':.2,'crop':0.}},
                       expected_label_sampling={label:{'positive':.3,'negative':.6,'unknown':.1} for label in CLASSES},
                       target_ranking={'weight':0., 'classes':TARGETS, 'helper_sha256':'helper',
                                       'scope':'original full known same domain', 'formula':'softplus group mean',
                                       'sampling_changed':False, 'new_labels_asserted':0, 'public_outputs_changed':False})
        treatment = copy.deepcopy(control)
        treatment['target_ranking']['weight'] = .25
        return control, treatment

    def history(self, active):
        rows = []
        for epoch in range(1, 7):
            rows.append({'epoch':epoch, 'domain_counts':{'dacl':10000,'damsegment':1400,'codebrim':2848},
                         'row_type_counts':{'full':10000,'crop':4248},
                         'full_target_joint_counts':{'dacl':{'00':3000,'10':1500,'01':1500,'11':152,'unknown':0},
                                                    'damsegment':{'00':300,'10':300,'01':300,'11':100,'unknown':0},
                                                    'codebrim':{'00':1000,'10':700,'01':700,'11':448,'unknown':0}},
                         'ranking_audit':{'unweighted_mean_batch_loss':.5 if active else 0.,
                                          'pair_count':12 if active else 0, 'contributing_batches':6 if active else 0,
                                          'pair_counts_by_domain_target':{'0:0':8,'1:1':4} if active else {}}})
        return {'actual_sampling_history':rows}

    def entries(self):
        entries = []
        for name, worst in ((REFERENCE,.25),(CONTROL,.24),(TREATMENT,.20)):
            entries.append({'run':name, 'worst_error':worst,
                            'per_class':{label:{'domains':{domain:{'fnr':worst,'fpr':.1} for domain in DOMAINS}} for label in TARGETS},
                            'ranking_ap':{domain:{label:{'ap':.8 if k in KNOWN_INDICES[domain] else None,
                                          'known_photos':COUNTS[domain] if k in KNOWN_INDICES[domain] else 0,
                                          'positive_photos':100 if k in KNOWN_INDICES[domain] else 0}
                                          for k,label in enumerate(CLASSES)} for domain in DOMAINS}})
        return entries

    def cache(self, domain):
        n = COUNTS[domain]
        # Actual saved torch targets are numeric JSON floats, including -1.0.
        targets = [[float(i % 2) if k in KNOWN_INDICES[domain] else -1. for k in range(7)] for i in range(n)]
        scores = [[.8 if target == 1 else .2 for target in row] for row in targets]
        raw = {'split':'val', 'weights_sha256':'checkpoint', 'targets':targets, 'probabilities':scores}
        frozen = {'signature':{'weights_sha256':'checkpoint','grid':1}, 'classes':CLASSES,
                  'probabilities':copy.deepcopy(scores)}
        entry = {'weights_sha256':'checkpoint', 'ranking_ap':{domain:{}}, 'per_class':{}}
        for k,label in enumerate(CLASSES):
            asserted = k in KNOWN_INDICES[domain]
            entry['ranking_ap'][domain][label] = {'ap':1. if asserted else None,
                    'known_photos':n if asserted else 0, 'positive_photos':n // 2 if asserted else 0}
            if label in TARGETS:
                entry['per_class'][label] = {'threshold':.5, 'domains':{domain:{'tp':n//2,'fn':0,'fp':0,'tn':n-n//2}}}
        return raw, frozen, entry

    def test_pair_requires_protocol_and_unchanged_source_sampling_and_truth(self):
        control,treatment = self.pair()
        self.assertEqual(validate_pair(control,treatment,INITIAL_SHA,self.protocol(),'protocol')['seed'],53)
        for mutation in ('source','unknown','weights','helper','protocol','ranking','source_extra','epochs','loss'):
            a,b = self.pair()
            if mutation == 'source':b['domain_proportions']=[.63,.09,.18,.1]
            if mutation == 'unknown':b['expected_label_sampling'][CLASSES[4]]['unknown']=0.
            if mutation == 'weights':b['pixel_positive_weights']='changed'
            if mutation == 'helper':b['target_ranking']['helper_sha256']='changed'
            if mutation == 'protocol':b['study_protocol_sha256']='changed'
            if mutation == 'ranking':b['target_ranking']['sampling_changed']=True
            if mutation == 'source_extra':b['train_convid']=192
            if mutation == 'epochs':b['actual_epochs']=5
            if mutation == 'loss':b['loss']='changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_pair(a,b,INITIAL_SHA,self.protocol(),'protocol')
        protocol = self.protocol();protocol['research_candidate_gate'] = {**GATE,'minimum_maximum_error_improvement_vs_reference_and_control':0.}
        with self.assertRaises(ValueError):validate_pair(*self.pair(),INITIAL_SHA,protocol,'protocol')

    def test_actual_draws_joint_states_and_ranking_use_all_epochs(self):
        self.assertTrue(validate_actual_pair(self.history(False),self.history(True))['treatment_ranking_used_all_six_epochs'])
        for mutation in ('joint','draw_budget','no_pairs','invalid_domain','unknown','control_pairs'):
            a,b = self.history(False),self.history(True)
            row = b['actual_sampling_history'][2]
            if mutation == 'joint':row['full_target_joint_counts']['dacl'].update({'00':2999,'10':1501})
            if mutation == 'draw_budget':row['domain_counts']['dacl']+=1
            if mutation == 'no_pairs':row['ranking_audit'].update(pair_count=0,contributing_batches=0,pair_counts_by_domain_target={})
            if mutation == 'invalid_domain':row['ranking_audit']['pair_counts_by_domain_target']={'3:0':12}
            if mutation == 'unknown':row['full_target_joint_counts']['dacl'].update({'00':2999,'unknown':1})
            if mutation == 'control_pairs':a['actual_sampling_history'][2]['ranking_audit']['pair_count']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_actual_pair(a,b)

    def test_nomination_requires_improvement_and_both_regression_guards(self):
        entries = self.entries()
        result = research_gate(*entries)
        self.assertTrue(result['research_candidate_nominated'])
        self.assertFalse(result['deployment_authorized'])
        for mutation in ('no_gain','target','other_ap','unknown_ap'):
            a,b,c = self.entries()
            if mutation == 'no_gain':c['worst_error']=.239
            if mutation == 'target':c['per_class'][TARGETS[0]]['domains']['damsegment']['fpr']=.121
            if mutation == 'other_ap':c['ranking_ap']['dacl'][CLASSES[3]]['ap']=.779
            if mutation == 'unknown_ap':c['ranking_ap']['damsegment'][CLASSES[4]]['ap']=.9
            with self.subTest(mutation=mutation):
                if mutation == 'unknown_ap':
                    with self.assertRaises(ValueError):research_gate(a,b,c)
                else:self.assertFalse(research_gate(a,b,c)['research_candidate_nominated'])
        a,b,c = self.entries();c['worst_error']=.235
        self.assertTrue(research_gate(a,b,c)['research_candidate_nominated'])

    def test_raw_cache_recalculates_ap_counts_and_keeps_unknowns_unknown(self):
        for domain in DOMAINS:
            raw,frozen,entry = self.cache(domain)
            measured,digest = verify_validation_cache(raw,frozen,entry,domain)
            self.assertEqual(len(digest),64)
            self.assertEqual(sum(point['ap'] is not None for point in measured.values()),len(KNOWN_INDICES[domain]))
        for mutation in ('count','ap','weights','split','order','unknown'):
            raw,frozen,entry = self.cache('damsegment')
            if mutation == 'count':entry['per_class'][TARGETS[0]]['domains']['damsegment']['fn']=1
            if mutation == 'ap':entry['ranking_ap']['damsegment'][TARGETS[0]]['ap']=.99
            if mutation == 'weights':raw['weights_sha256']='wrong'
            if mutation == 'split':raw['split']='test'
            if mutation == 'order':frozen['probabilities'].reverse()
            if mutation == 'unknown':raw['targets'][0][4]=0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_validation_cache(raw,frozen,entry,'damsegment')

    def test_grouped_ties_ap_and_wilson_bounds_are_numeric_not_accuracy_claims(self):
        self.assertAlmostEqual(average_precision([1,0,1,0],[.9,.9,.5,.2]),7/12)
        self.assertIsNone(average_precision([0,0],[.7,.1]))
        zero = descriptive_interval(0,100)
        self.assertAlmostEqual(zero[0],0.)
        self.assertAlmostEqual(zero[1],.03699349820698568)
        all_errors = descriptive_interval(100,100)
        self.assertAlmostEqual(all_errors[0],1-zero[1])
        self.assertAlmostEqual(all_errors[1],1.)
        for pair in ((-1,10),(11,10),(0,0),(True,10),(1,True)):
            with self.assertRaises(ValueError):descriptive_interval(*pair)

    def test_negative_audit_cannot_turn_unknown_into_normal(self):
        training,_ = self.pair()
        training['auxiliary_manifest_sha256']='aux'
        audit = {'status':'audited','classes':CLASSES,'full_base_rows':14248,'derived_crop_rows':12041,
                 'input_sha256':{'core_spatial_train':'core','auxiliary_train':'aux'},
                 'policy':{key:False for key in ('native_validation_or_test_annotations_opened',
                           'validation_or_test_labels_used','validation_or_test_images_read','model_predictions_used',
                           'source_annotations_or_manifests_modified','training_executed')},
                 'counts':{'full':{'general_normal_or_safe_ground_truth_asserted':False,
                           'both_targets_known_negative':5915,
                           'per_label':{TARGETS[0]:{'negative':9387},TARGETS[1]:{'negative':9460}}}}}
        self.assertFalse(validate_train_audit(audit,training)['global_normal_or_safe_ground_truth_asserted'])
        audit['counts']['full']['general_normal_or_safe_ground_truth_asserted']=True
        with self.assertRaises(ValueError):validate_train_audit(audit,training)

    def test_technical_check_is_frozen_shape_proof_not_field_accuracy(self):
        entries = [{'run':REFERENCE,'weights_sha256':'initializer'},
                   {'run':CONTROL,'weights_sha256':'control'},
                   {'run':TREATMENT,'weights_sha256':'treatment'}]
        proof = {'status':'passed','study_protocol_sha256':'protocol','app_profile_sha256':'profile',
                 'app_profile_unchanged':True,'research_models_deployed':False,
                 'accuracy_measured_by_this_check':False,'independent_field_safety_verified':False,
                 'experiments':[{'run':entry['run'],'weights_sha256':entry['weights_sha256'],
                     'study_protocol_sha256':'protocol','public_shape':[1,7],
                     'private_spatial_shape':[1,7,80,80],'auxiliary_shape':[1,19],
                     'outputs_finite':True,'public_training_logits_equal':True} for entry in entries[1:]],
                 'actual_train_gpu_preflight':{'status':'passed','finite_loss_and_gradients':True,
                                              'public_shape':[8,7],'ranking_pair_count':32},
                 'pre_training_source_snapshot':{'comparison_started_before_outcomes':True,
                     'pre_training_commit':'frozen','sources':{path:{'equal':True,
                         'git_blob_sha256':'source','executed_file_sha256':'source'} for path in (
                         'scripts/train_facility_spatial.py','scripts/facility_target_ranking.py',
                         'reports/facility-target-discrimination-protocol.json')}}}
        with patch('scripts.report_facility_discrimination.sha', return_value='source'):
            result = validate_technical_verification(proof,entries,'protocol','profile')
            self.assertFalse(result['accuracy_measured_by_technical_check'])
            for mutation in ('checkpoint','protocol','outputs','source','field_claim','preflight'):
                changed = copy.deepcopy(proof)
                if mutation == 'checkpoint':changed['experiments'][0]['weights_sha256']='other'
                if mutation == 'protocol':changed['study_protocol_sha256']='changed'
                if mutation == 'outputs':changed['experiments'][1]['public_shape']=[1,8]
                if mutation == 'source':changed['pre_training_source_snapshot']['sources']['scripts/train_facility_spatial.py']['git_blob_sha256']='changed'
                if mutation == 'field_claim':changed['independent_field_safety_verified']=True
                if mutation == 'preflight':changed['actual_train_gpu_preflight']['finite_loss_and_gradients']=False
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    validate_technical_verification(changed,entries,'protocol','profile')


if __name__ == '__main__':
    unittest.main()
