from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from safelog_ai.review_feedback import canonical_sha256
from safelog_ai.human_training_feedback import build_overlay,validate_submission,SCHEMA
from scripts import serve_facility_human_review as server
from scripts.facility_human_training_dataset import human_exposure_draws,HumanReviewPhotos
from scripts import train_facility_context as original_dataset
from safelog_ai.auxiliary_classifier import AUX_CLASSES
from scripts.prepare_facility_human_review import SOURCE_BUDGETS,COUNT,select_cases


def package():
    case={"case_id":"case-1","domain":"dacl","image":"data/a.jpg","image_sha256":"b"*64,"original_targets":[1,0,0,0,0,0,0],"probabilities":[.2,.6,.1,.1,.1,.1,.1]}
    doc={"schema":"facility_train_review_v1","split":"train","classes":server.validate_package.__globals__["CLASSES"],"weights_sha256":"a"*64,"cases":[case]}
    doc["package_content_sha256"]=canonical_sha256(doc);return doc


def submission(p,judgements):
    draft=server.blank_draft(p);draft["cases"][0]["judgements"]=judgements
    stamp=datetime.now(timezone.utc).isoformat()
    draft["cases"][0]["reviewed_at"]=[None if j=="unreviewed"else stamp for j in judgements]
    return {"schema":SCHEMA,"purpose":"human_reviewed_train_photo_labels","source_package_sha256":p["package_content_sha256"],
            "feedback":server.to_feedback(draft,p),"submitted_for_training":True,"test_fixture":False}


class HumanFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)
    def core(self):
        p=package()
        return {"split":"train","classes":p["classes"],"full_count":1,"items":[{"image":"data/a.jpg","domain":"dacl","targets":[1,0,0,0,0,0,0]},
            {"image":"data/crop.jpg","parent_image":"data/a.jpg","domain":"dacl","targets":[1,0,1,1,1,1,1]}]}
    def test_changed_labels_disable_pixel_aux_and_child_without_inventing_masks(self):
        p=package();core=self.core();old=deepcopy(core);overlay,counts=build_overlay(submission(p,["absent","present"]),p,core)
        self.assertEqual(core,old);self.assertEqual(overlay["items"][0]["targets"][:2],[0,1])
        self.assertEqual(overlay["items"][1]["targets"][:2],[-1,-1]);self.assertEqual(overlay["items"][1]["targets"][2:],[1]*5)
        self.assertEqual(overlay["items"][0]["human_pixel_unknown_columns"],[0,1]);self.assertEqual(overlay["items"][0]["human_auxiliary_unknown_tags"],["Crack","Spalling"])
        self.assertEqual(counts["changed_full_photo_targets"],2);self.assertFalse(counts["expert_qualification_verified"])
    def test_uncertain_disables_targets_but_unreviewed_keeps_original(self):
        p=package();overlay,counts=build_overlay(submission(p,["uncertain","absent"]),p,self.core())
        self.assertEqual(overlay["items"][0]["targets"][:2],[-1,0]);self.assertEqual(overlay["items"][1]["targets"][:2],[-1,0])
        untouched,_=build_overlay(submission(p,["unreviewed","absent"]),p,self.core())
        self.assertEqual(untouched["items"][0]["targets"][0],1)
    def test_same_known_labels_preserve_original_supervision(self):
        p=package();core=self.core();overlay,counts=build_overlay(submission(p,["present","absent"]),p,core)
        self.assertEqual(overlay["items"],core["items"]);self.assertEqual(counts["changed_full_photo_targets"],0)
    def test_no_definite_input_draft_ai_and_fixture_cannot_train(self):
        p=package()
        for choices in(["unreviewed","unreviewed"],["uncertain","uncertain"]):
            with self.assertRaises(ValueError):validate_submission(submission(p,choices),p)
        for key,value in(("submitted_for_training",False),("test_fixture",True),("schema","facility_ai_observations_v1")):
            doc=submission(p,["present","absent"]);doc[key]=value
            with self.assertRaises(ValueError):validate_submission(doc,p)
    def test_stale_and_wrong_package_rejected(self):
        p=package();doc=submission(p,["present","absent"]);doc["source_package_sha256"]="c"*64
        with self.assertRaises(ValueError):build_overlay(doc,p,self.core())
        core=self.core();core["items"][0]["targets"][0]=0
        with self.assertRaises(ValueError):build_overlay(submission(p,["present","absent"]),p,core)
    def test_bounded_exposure_keeps_domain_crop_positions_and_original_arrays(self):
        items=[{"image":"a","domain":"dacl","targets":[1,0]},{"image":"b","domain":"codebrim","targets":[0,1]},
               {"image":"crop","domain":"dacl","targets":[1,1]}]
        original=np.tile(np.resize(np.array([0,1,2],np.int64),14248),(6,1));snapshot=original.copy()
        draws,counts=human_exposure_draws(original,items,2,{"a":[0,1],"b":[0,1]})
        self.assertTrue(np.array_equal(original,snapshot));self.assertTrue(np.array_equal(draws[original==2],original[original==2]))
        for epoch in range(6):self.assertEqual(counts[epoch],{"dacl":2,"codebrim":2})
        with self.assertRaises(ValueError):human_exposure_draws(original,items,2,{"crop":[0]})
    def test_selection_budget_is_exact_and_sampling_repeatable(self):
        self.assertEqual(sum(sum(v)for v in SOURCE_BUDGETS.values()),COUNT)
        items=[];scores={}
        for source,(errors,correct)in SOURCE_BUDGETS.items():
            for i in range(errors+correct+30):
                name=f"{source}/{i}";items.append({"image":name,"domain":source,"targets":[0,0]+[0]*5})
                scores[name]=[.9 if i<errors+10 else .1,.1]+[.1]*5
        a,stats=select_cases(items,scores,{"concrete_crack":.5,"concrete_spalling":.5})
        b,_=select_cases(items,scores,{"concrete_crack":.5,"concrete_spalling":.5})
        self.assertEqual(a,b);self.assertEqual(len(a),200);self.assertEqual(sum(r["sampled"]for r in stats if r["stratum"]=="error_candidate"),120)
    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in(b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):server.strict_json(raw)
    def test_actual_dataset_disables_changed_supervision_after_shared_augmentation(self):
        with tempfile.TemporaryDirectory()as directory,patch.object(original_dataset,"ROOT",Path(directory)):
            root=Path(directory);Image.new("RGB",(96,96),"gray").save(root/"image.jpg")
            np.savez(root/"mask.npz",mask=np.ones((7,80,80),np.uint8),known=np.ones(7,np.uint8))
            rows=[{"image":"image.jpg","pixel_target":"mask.npz","domain":"dacl","targets":[0,1,0,0,0,0,0],
                   "human_pixel_unknown_columns":[0],"human_auxiliary_unknown_tags":["Crack"]}]
            dataset=HumanReviewPhotos(rows,{"image.jpg":[1]*19},1,{"dacl":0},640);sample=dataset[0]
            self.assertEqual(float(sample[5][0]),0);self.assertEqual(float(sample[5][1]),1)
            self.assertEqual(float(sample[7][AUX_CLASSES.index("Crack")]),0);self.assertEqual(float(sample[7][AUX_CLASSES.index("Spalling")]),1)
            self.assertEqual(float(sample[4][0].sum()),6400);self.assertEqual(sample[1][:2].tolist(),[0,1])


if __name__=="__main__":unittest.main()
