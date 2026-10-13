import unittest
import torch
from safelog_ai.belt_fine_tiles import fine_boxes, fuse, policy, guards, MODES


class FineTilesTests(unittest.TestCase):
    def test_coverage_orientation_and_bound(self):
        for width, height in [(1920, 1080), (3840, 2160), (2160, 3840), (3456, 3456), (700, 250), (3000, 1000)]:
            boxes = fine_boxes(width, height)
            self.assertLessEqual(len(boxes), 15)
            self.assertTrue(boxes)
            for l, t, r, b in boxes:
                self.assertTrue(0 <= l < r <= width and 0 <= t < b <= height)
                self.assertEqual(r-l, b-t)
            # All interval cells between edges must be covered, including seams.
            xs = sorted({0, width, *(x for b in boxes for x in (b[0], b[2]))})
            ys = sorted({0, height, *(y for b in boxes for y in (b[1], b[3]))})
            for x1, x2 in zip(xs, xs[1:]):
                for y1, y2 in zip(ys, ys[1:]):
                    x, y = (x1+x2)/2, (y1+y2)/2
                    self.assertTrue(any(l <= x < r and t <= y < b for l, t, r, b in boxes))
            self.assertEqual(set(fine_boxes(height, width)), {(t,l,b,r) for l,t,r,b in boxes})
        self.assertEqual(len(fine_boxes(3840, 2160)), 15)
        self.assertEqual(fine_boxes(640, 360), [])
        self.assertEqual(fine_boxes(8000, 1000), [])
        with self.assertRaises(ValueError): fine_boxes(0, 100)

    def test_fusion_keeps_padding_and_fallback_unchanged(self):
        g = torch.full((1,1,10,10), .2); c = torch.full_like(g,.6); f = torch.full_like(g,.8)
        for mode, value in zip(MODES, [.4,.5,.45]):
            out = fuse(g,c,f,(1,2,8,5),mode)
            self.assertTrue(torch.allclose(out[:,:,2:7,1:9], torch.full((1,1,5,8),value)))
            self.assertTrue(torch.equal(out[:,:,:2],g[:,:,:2]))
            self.assertTrue(torch.equal(fuse(g,None,None,(0,0,10,10),mode),g))
        self.assertTrue(torch.equal(g,torch.full_like(g,.2)))

    def test_policy_has_distinct_identity_and_no_annotation_input(self):
        self.assertEqual(len({policy(m)['sha256'] for m in MODES}),3)
        for m in MODES:
            self.assertFalse(policy(m)['annotations_at_inference'])
            self.assertEqual(policy(m)['model_count'],1)
        with self.assertRaises(ValueError): policy('guess')

    def test_selection_rejects_small_damage_regression(self):
        b = {'pixel_iou':.476,'pixel_fnr':.329,'pixel_fpr':.0069,'photo_rates':{'fnr':.00566},'positive_photos_without_any_gt_overlap':28}
        slices={'GT-area:<=0.25%':{'pixel_iou':.114,'pixel_fnr':.48},'sequence:a':{'pixel_fnr':.46}}
        r=dict(b,pixel_iou=.50,pixel_fnr=.3,pixel_fpr=.006)
        s={**slices,'GT-area:<=0.25%':{'pixel_iou':.10,'pixel_fnr':.49}}
        self.assertFalse(all(guards(r,b,s,slices).values()))
        self.assertTrue(all(guards(r,b,slices,slices).values()))


if __name__ == '__main__': unittest.main()
