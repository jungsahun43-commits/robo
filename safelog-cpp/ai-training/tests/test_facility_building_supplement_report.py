import copy
import unittest

from scripts.report_facility_building_supplement import (
    CLASSES, DOMAINS, MATCHED_KEYS, INITIAL_SHA,
    validate_building_pair, validate_sampling, validate_actual_draws,
)


class BuildingSupplementReportTests(unittest.TestCase):
    def pair(self):
        control = {key: 'fixed' for key in MATCHED_KEYS}
        control.update(status='complete', architecture='lraspp_mobilenet_facility_auxiliary_v1', classes=CLASSES,
                       seed=52, requested_epochs=6, actual_epochs=6, patience=6, imgsz=640,
                       draws_per_epoch=14248, batch_size=8, backbone_lr=.00004, head_lr=.00025,
                       auxiliary_weight=.5, validation_domains=DOMAINS, initial_weights_sha256=INITIAL_SHA,
                       train_dacl=6225, train_damsegment=1585, train_codebrim=6438, val_dacl=710, val_damsegment=424,
                       spatial_manifest_sha256='core', core_spatial_manifest_sha256='core', domain_proportions=[.7, .1, .2],
                       spatial_manifest_audit={'full_photo_or_patch_rows':14248,'per_label_pixel_cells':{'a':{'positive':1,'negative':1}}},
                       expected_sampling={'dacl':{'full':.4,'crop':.3},'damsegment':{'full':.07,'crop':.03},'codebrim':{'full':.2,'crop':0}},
                       expected_label_sampling={label:{'positive':.3,'negative':.6,'unknown':.1} for label in CLASSES},
                       pixel_positive_weights=[1.]*7, auxiliary_positive_weights=[1.]*19,
                       photo_positive_weights={domain:[1.]*7 for domain in DOMAINS})
        treatment = copy.deepcopy(control)
        treatment.update(train_convid=200, photo_supplement_sha256='a'*64, domain_proportions=[.63,.09,.18,.10])
        treatment['expected_sampling'] = {d:{k:.9*v for k,v in mass.items()} for d,mass in control['expected_sampling'].items()}
        treatment['expected_sampling']['convid'] = {'full':.1,'crop':0}
        treatment['photo_positive_weights']['convid'] = [.2]*7
        treatment['expected_label_sampling'] = {label:{'positive':.27,'negative':.54,'unknown':.19} for label in CLASSES}
        for label in CLASSES[:2]: treatment['expected_label_sampling'][label] = {'positive':.32,'negative':.54,'unknown':.14}
        return control, treatment

    def test_matched_pair_accepts_intentional_new_domain_and_fixed_budget(self):
        control, treatment = self.pair()
        result = validate_building_pair(control, treatment, INITIAL_SHA)
        self.assertEqual(result['treatment_convid_photo_rows'],200)
        self.assertTrue(result['original_conditional_full_crop_sampling_preserved'])
        self.assertNotEqual(result['control_domain_proportions'],result['treatment_domain_proportions'])

    def test_missing_running_or_changed_configuration_is_rejected(self):
        for key, value in (('status','running'),('seed',53),('draws_per_epoch',14748),
                           ('head_lr',.001),('spatial_manifest_sha256','different'),('train_convid',201)):
            control,treatment=self.pair();treatment[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_building_pair(control,treatment,INITIAL_SHA)
        control,treatment=self.pair();treatment.pop('auxiliary_manifest_sha256')
        with self.assertRaises(ValueError):validate_building_pair(control,treatment,INITIAL_SHA)
        with self.assertRaises(ValueError):validate_building_pair(*self.pair(),'different')

    def test_crop_conditional_weights_and_unknown_labels_must_remain(self):
        for mutation in ('crop','unknown','target_negative','pixel','pixel_weight','auxiliary_weight','photo_weight','source_photo_weight','batch','helper_hash'):
            control,treatment=self.pair()
            if mutation=='crop':treatment['expected_sampling']['dacl'].update(full=.4,crop=.23)
            if mutation=='unknown':treatment['expected_label_sampling'][CLASSES[2]].update(negative=.64,unknown=.09)
            if mutation=='target_negative':treatment['expected_label_sampling'][CLASSES[0]].update(negative=.59,unknown=.09)
            if mutation=='pixel':treatment['spatial_manifest_audit']['per_label_pixel_cells']['a']['positive']=2
            if mutation=='pixel_weight':treatment['pixel_positive_weights'][0]=2.
            if mutation=='auxiliary_weight':treatment['auxiliary_positive_weights'][0]=2.
            if mutation=='photo_weight':treatment['photo_positive_weights']['dacl'][0]=2.
            if mutation=='source_photo_weight':treatment['photo_positive_weights']['convid'][0]=1.
            if mutation=='batch':treatment['batch_size']=16
            if mutation=='helper_hash':treatment['photo_supplement_helper_sha256']='different'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_building_pair(control,treatment,INITIAL_SHA)

    def test_six_actual_epoch_draws_required_and_supplement_must_be_used(self):
        entry={'actual_sampling_history':[{'epoch':i,'domain_counts':{'dacl':10400,'damsegment':1000,'codebrim':2848},'row_type_counts':{'full':10000,'crop':4248}} for i in range(1,7)]}
        validate_actual_draws(entry)
        treatment=copy.deepcopy(entry)
        for row in treatment['actual_sampling_history']:row['domain_counts'].update(dacl=9400,convid=1000)
        validate_actual_draws(treatment,treatment=True)
        treatment['actual_sampling_history'][0]['domain_counts'].update(dacl=10400,convid=0)
        with self.assertRaises(ValueError):validate_actual_draws(treatment,treatment=True)
        entry['actual_sampling_history'].pop()
        with self.assertRaises(ValueError):validate_actual_draws(entry)


if __name__=='__main__':unittest.main()
