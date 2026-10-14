"""Use real-mask training for photo presence and preserve original locations."""
from .ksdd_supervised import KSDDSupervisedMain, CATEGORY, ksdd_photo_score

VERSION = 'safelog-ksdd-supervised-photo-only-v11'


class KSDDPhotoMain(KSDDSupervisedMain):
    def forward_ksdd_photo(self, original256, native, valid):
        original_location = super().forward_equipment(original256, CATEGORY)
        supervised_probability = self.forward_ksdd(native).sigmoid()
        return original_location, supervised_probability, ksdd_photo_score(supervised_probability, valid)
