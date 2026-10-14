"""Synthetic integrity cases; none of these observations are real labels."""
import copy
import json
import unittest
from safelog_ai.loco_relation_review import canonical_sha, strict_json, validate_observations, review_summary


def fixture():
    def shape(i, x):
        return {'shape_id': i, 'label': 'connector', 'points': [[x,.3],[x+.1,.3],[x+.1,.5],[x,.5]]}
    case = {'id': 'mvtec_loco/splicing_connectors/train/good/synthetic.png',
            'phase': 'fit', 'publisher_split': 'train', 'appearance_group_proxy': 'synthetic',
            'label': 0, 'masks_used': 0, 'image_sha256': 'a'*64, 'annotation_sha256': 'b'*64,
            'connector_shapes': [shape(1,.1), shape(2,.8)],
            'cable_shapes': [{'shape_id': 3, 'label': 'cable', 'points': [[.2,.4],[.8,.4],[.8,.42]]}],
            'proposed_groups': [{'shape_id': 1, 'side': 'left'}, {'shape_id': 2, 'side': 'right'}]}
    package = {'schema': 'loco_relation_review_package_v1', 'training_ready': False, 'cases': [case]}
    package['package_sha256'] = canonical_sha(package)
    row = {'id': case['id'], 'image_sha256': case['image_sha256'], 'annotation_sha256': case['annotation_sha256'],
           'status': 'confirmed', 'reviewed_utc': '2026-10-15T00:00:00.000Z', 'polygons_accepted': True,
           'groups': case['proposed_groups'], 'left_count': 1, 'right_count': 1, 'wire_count': 1,
           'wire_color': 'blue', 'left_endpoint': [.2,.4], 'right_endpoint': [.8,.4],
           'left_pin': 1, 'right_pin': 1, 'note': 'Synthetic test only; not an observed label.'}
    doc = {'schema': 'loco_relation_observations_v1', 'package_sha256': package['package_sha256'],
           'reviewer': 'synthetic_test_not_real_review', 'exported_utc': '2026-10-15T00:00:01.000Z', 'cases': [row]}
    return package, doc


class RelationReviewTests(unittest.TestCase):
    def test_confirmation_never_creates_training_labels(self):
        p,d=fixture(); before=json.dumps(d)
        self.assertIs(validate_observations(d,p),d)
        s=review_summary(d,p)
        self.assertEqual(s['review_counts']['confirmed'],1)
        self.assertFalse(s['training_ready']);self.assertEqual(s['training_labels_created'],0)
        self.assertEqual(json.dumps(d),before)

    def test_source_tamper_duplicate_TEST_and_CAL_reject(self):
        changes = [lambda d: d['cases'][0].update(image_sha256='c'*64),
                   lambda d: d['cases'][0].update(annotation_sha256='c'*64),
                   lambda d: d['cases'].append(copy.deepcopy(d['cases'][0])),
                   lambda d: d['cases'][0].update(id='mvtec_loco/splicing_connectors/test/good/000.png'),
                   lambda d: d['cases'][0].update(id='unknown/cal/000.png'),
                   lambda d: d.update(package_sha256='c'*64)]
        for change in changes:
            p,d=fixture();change(d);before=copy.deepcopy(d)
            with self.assertRaises(ValueError):validate_observations(d,p)
            self.assertEqual(d,before)
        p,d=fixture();p['cases'][0]['id']='source/test/good/000.png'
        p['package_sha256']=canonical_sha({k:v for k,v in p.items() if k!='package_sha256'})
        d['package_sha256']=p['package_sha256'];d['cases'][0]['id']=p['cases'][0]['id']
        with self.assertRaises(ValueError):validate_observations(d,p)

    def test_incomplete_count_endpoint_and_group_cannot_confirm(self):
        changes = [lambda r:r.update(polygons_accepted=False), lambda r:r.update(left_count=2),
                   lambda r:r.update(left_pin=2), lambda r:r.update(left_endpoint=None),
                   lambda r:r.update(left_endpoint=[.99,.99]), lambda r:r.update(left_endpoint=[float('nan'),.4]),
                   lambda r:r.update(wire_color='unknown'), lambda r:r.update(wire_count=True),
                   lambda r:r.update(groups=[{'shape_id':1,'side':'left'},{'shape_id':1,'side':'right'}])]
        for change in changes:
            p,d=fixture();change(d['cases'][0])
            with self.assertRaises(ValueError):validate_observations(d,p)

    def test_uncertain_stays_uncertain_and_partial_drafts_valid(self):
        p,d=fixture();r=d['cases'][0];r.update(status='uncertain',left_endpoint=None,wire_color='unknown')
        self.assertEqual(review_summary(d,p)['review_counts']['uncertain'],1)
        r.update(status='draft',reviewed_utc=None)
        self.assertEqual(review_summary(d,p)['review_counts']['draft'],1)
        d['cases']=[];self.assertEqual(review_summary(d,p)['missing_cases'],1)

    def test_strict_json_and_calendar_reject(self):
        for raw in ('{"x":1,"\\u0078":2}','{"x":NaN}','{"x":1e999}','{} {}','\ufeff{}'):
            with self.assertRaises((ValueError,json.JSONDecodeError)):strict_json(raw)
        for stamp in ('2026-02-30T00:00:00.000Z','2026-10-15T00:00:60.000Z','2026-10-15T00:00:00Z','2026-10-15T00:00:02.000Z'):
            p,d=fixture();d['cases'][0]['reviewed_utc']=stamp
            with self.assertRaises(ValueError):validate_observations(d,p)


if __name__=='__main__':unittest.main()
