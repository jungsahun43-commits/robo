"""Study boundary checks: provenance mutations cannot become a training pair."""
from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from scripts import facility_native_roi_study as study
from scripts.preflight_facility_native_roi import replay_indices


class NativeRoiProtocolTests(TestCase):
    def protocol(self):
        p=study.fixed_values()
        p['source_sha256']={path:'a'*64 for path in study.SOURCE_FILES}
        p['paired_manifest_sha256']={v:'a'*64 for v in study.MANIFESTS}
        for key in ('core_spatial_manifest_sha256','auxiliary_manifest_sha256',
                    'paired_data_audit_sha256','app_profile_sha256','source_records_sha256',
                    'source_preparation_sha256'):
            p[key]='a'*64
        return p

    def validate(self,p):
        def digest(path):
            return study.INITIAL_SHA if Path(path).name=='best.pt' else 'a'*64
        with patch.object(study,'sha',side_effect=digest):
            return study.validate_protocol(p,Path('/temporary-study'))

    def test_valid_fixed_pair_preserves_input_contract(self):
        p=self.protocol();actual=self.validate(p)
        self.assertEqual(actual['imgsz_by_variant'],{'control':640,'native':640})
        self.assertEqual(actual['architecture_by_variant']['control'],actual['architecture_by_variant']['native'])
        self.assertIsNot(actual,p)

    def test_budget_or_holdout_mutation_is_rejected(self):
        for key,value in [('requested_epochs',7),('seed',57),('batch_size',16),
                          ('source_test_inference_executed',True),('declared_before_training',False),
                          ('app_model_promoted',True)]:
            with self.subTest(key=key):
                p=self.protocol();p[key]=value
                with self.assertRaises(ValueError):self.validate(p)

    def test_runtime_bytes_cannot_drift(self):
        p=self.protocol();p['source_sha256'][study.SOURCE_FILES[0]]='b'*64
        with self.assertRaisesRegex(ValueError,'runtime source'):self.validate(p)

    def test_missing_source_inventory_is_rejected(self):
        p=self.protocol();p['source_sha256'].pop(study.SOURCE_FILES[-1])
        with self.assertRaisesRegex(ValueError,'inventory'):self.validate(p)

    def test_native_manifest_and_original_supervision_are_bound(self):
        for key in ('native','control'):
            p=self.protocol();p['paired_manifest_sha256'][key]='b'*64
            with self.assertRaisesRegex(ValueError,'manifest bytes'):self.validate(p)
        p=self.protocol();p['core_spatial_manifest_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'Protected'):self.validate(p)

    def test_recipe_and_gate_cannot_be_changed_after_declaration(self):
        for key in ('native_roi_recipe','research_candidate_gate'):
            p=self.protocol();p[key]=deepcopy(p[key]);p[key]['new_labels' if key=='native_roi_recipe' else 'minimum_improvement_pp']=1
            with self.assertRaises(ValueError):self.validate(p)


class TrainReplayCoverageTests(TestCase):
    def fixture(self):
        rows=[{'image':f'data/full{i}.png','domain':domain,'targets':[0]*7}
              for i,domain in enumerate(('dacl','damsegment','codebrim'))]
        rows+=[{'image':f'data/crop{i}.png','domain':'dacl',
                'targets':target+[0]*5} for i,target in enumerate(([0,0],[1,0],[-1,1],[0,0],[0,1]))]
        current=deepcopy(rows)
        for i in (3,4,5):current[i]['image']=f'data/native{i}.png'
        return {'original_items':rows,'items':current}

    def test_replay_includes_all_sources_negative_positive_unknown_and_unchanged(self):
        data=self.fixture();indices=replay_indices(data,3)
        self.assertEqual(len(set(indices)),8)
        self.assertTrue({0,1,2,3,4,5}.issubset(indices))
        self.assertTrue(any(data['original_items'][i]['image']==data['items'][i]['image'] for i in indices))

    def test_unprepared_pair_cannot_pass_replay(self):
        data=self.fixture();data['items']=deepcopy(data['original_items'])
        with self.assertRaises(ValueError):replay_indices(data,3)
