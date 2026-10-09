"""AI-curation safeguards and exact exposure invariants, using synthetic rows."""
import copy
import unittest
import numpy as np
from scripts.facility_ai_agreement import select_agreement,agreement_draws

class AgreementTests(unittest.TestCase):
    def fixture(self):
        cases=[{"case_id":str(i),"image":"p"+str(i),"image_sha256":"a"*64,"annotation":{"sha256":"b"*64},"original_targets":[0,0,-1,-1,-1,-1,-1]}for i in range(200)]
        notes={"schema":"facility_ai_visual_selection_v1","observer_type":"ai","expert_confirmed":False,"human_feedback_used":False,"split":"train",
            "cases":[{"case_id":r["case_id"],"image_sha256":r["image_sha256"],"annotation_sha256":r["annotation"]["sha256"],"contact_photo_viewed":True,"evidence":"No visible surface indicator in this view","judgements":["absent","absent"]}for r in cases]}
        return notes,cases
    def data(self):
        items=[{"image":str(i),"domain":"dacl"if i<10 else"codebrim","targets":[0,0,1,-1,-1,-1,-1]}for i in range(24)]
        draws=np.tile(np.arange(14248,dtype=np.int64)%24,(6,1));return items,draws
    def test_ai_identity_not_human(self):
        n,c=self.fixture();eligible,audit=select_agreement(n,c);self.assertEqual(len(eligible),200);self.assertEqual(audit,{"both_definite_agreement_photo":200})
    def test_uncertain_excluded(self):
        n,c=self.fixture();n["cases"][0]["judgements"][0]="uncertain";e,a=select_agreement(n,c);self.assertNotIn("p0",e);self.assertEqual(a["uncertain_photo"],1)
    def test_disagreement_excluded(self):
        n,c=self.fixture();n["cases"][0]["judgements"][0]="present";e,a=select_agreement(n,c);self.assertNotIn("p0",e);self.assertEqual(a["definite_disagreement_photo"],1)
    def test_fake_expert_rejected(self):
        n,c=self.fixture();n["expert_confirmed"]=True
        with self.assertRaises(ValueError):select_agreement(n,c)
    def test_fake_human_rejected(self):
        n,c=self.fixture();n["human_feedback_used"]=True
        with self.assertRaises(ValueError):select_agreement(n,c)
    def test_unviewed_rejected(self):
        n,c=self.fixture();n["cases"][0]["contact_photo_viewed"]=False
        with self.assertRaises(ValueError):select_agreement(n,c)
    def test_identity_tampering_rejected(self):
        n,c=self.fixture();n["cases"][0]["image_sha256"]="f"*64
        with self.assertRaises(ValueError):select_agreement(n,c)
    def test_duplicates_rejected(self):
        n,c=self.fixture();n["cases"][1]=copy.deepcopy(n["cases"][0])
        with self.assertRaises(ValueError):select_agreement(n,c)
    def test_bounded_draws_preserve_targets_source_crops(self):
        items,draws=self.data();original=draws.copy();candidate,counts=agreement_draws(draws,items,20,["1","12"])
        self.assertEqual(counts,[4]*6);self.assertTrue(np.array_equal(draws,original))
        for before,after in zip(draws,candidate):
            self.assertTrue(np.array_equal(before[before>=20],after[before>=20]))
            for i in(1,12):self.assertEqual(int((after==i).sum())-int((before==i).sum()),2)
            for i,j in zip(before,after):self.assertEqual((items[i]["domain"],items[i]["targets"]),(items[j]["domain"],items[j]["targets"]))
    def test_non_full_rejected(self):
        items,draws=self.data()
        with self.assertRaises(ValueError):agreement_draws(draws,items,20,["23"])
    def test_duplicate_selected_rejected(self):
        items,draws=self.data()
        with self.assertRaises(ValueError):agreement_draws(draws,items,20,["1","1"])
    def test_non_integer_draws_rejected(self):
        items,draws=self.data()
        with self.assertRaises(ValueError):agreement_draws(draws.astype(np.float64),items,20,["1"])

if __name__=="__main__":unittest.main()
