import unittest
import evaluate_ablation_mymethod as a
class TestAblationWrapper(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(set(a.VARIANTS), {'eviction_only','anchor_only','full_no_count','full'})
        self.assertEqual(a._variant_group_mode('anchor_only'),'anchor_only')
        self.assertEqual(a._variant_group_mode('full_no_count'),'full')
if __name__=='__main__': unittest.main()
