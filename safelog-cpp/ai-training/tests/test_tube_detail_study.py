import unittest
from safelog_ai.tube_detail_study import tube_adaptation_rows
from safelog_ai.source_detail_study import photo_bucket_means


class TubeAdaptationTests(unittest.TestCase):
    def test_only_audited_tube_normal_and_source_are_admitted(self):
        def row(key, phase, category, active=True):
            return {'photo_id':key,'phase':phase,'category':category,'is_NEU':False,'active_support':True,
                    'training_active':phase=='fit' and active,'validation_active':phase=='validation' and active,
                    'label':0,'positive_pixels':0,'bucket':category}
        result=tube_adaptation_rows([row('a','fit','mpdd_tubes'),row('b','validation','mpdd_tubes'),
            row('c','fit','mpdd_tubes',False),row('black','fit','mpdd_bracket_black'),row('cal','cal','mpdd_tubes')])
        self.assertEqual([r['photo_id'] for r in result['fit']],['a'])
        self.assertEqual([r['photo_id'] for r in result['validation']],['b'])

    def test_three_validation_groups_preserve_original_photo_weight(self):
        rows=[{'photo_id':p,'bucket':b} for p,b in [('a','real_positive'),('a','real_positive'),
              ('b','real_positive'),('a','annotation_negative'),('tube','mpdd_tubes')]]
        self.assertEqual(photo_bucket_means(rows,[1,3,10,4,5]),{'real_positive':6,'annotation_negative':4,'mpdd_tubes':5})


if __name__=='__main__':unittest.main()
