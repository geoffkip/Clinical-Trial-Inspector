import unittest
from modules.utils import normalize_sponsor, get_sponsor_variations, SPONSOR_MAPPINGS

class TestSponsorNormalization(unittest.TestCase):
    def test_normalize_sponsor_aliases(self):
        """Test that common aliases map to canonical names."""
        self.assertEqual(normalize_sponsor("J&J"), "Janssen")
        self.assertEqual(normalize_sponsor("Johnson & Johnson"), "Janssen")
        self.assertEqual(normalize_sponsor("GSK"), "GlaxoSmithKline")
        self.assertEqual(normalize_sponsor("Merck"), "Merck Sharp & Dohme")
        self.assertEqual(normalize_sponsor("BMS"), "Bristol-Myers Squibb")

    def test_normalize_sponsor_variations(self):
        """Test that specific DB variations map to canonical names."""
        self.assertEqual(normalize_sponsor("Janssen Research & Development, LLC"), "Janssen")
        self.assertEqual(normalize_sponsor("Pfizer Inc."), "Pfizer")
        self.assertEqual(normalize_sponsor("Merck Sharp & Dohme LLC"), "Merck Sharp & Dohme")

    def test_normalize_sponsor_canonical(self):
        """Test that canonical names return themselves."""
        self.assertEqual(normalize_sponsor("Janssen"), "Janssen")
        self.assertEqual(normalize_sponsor("Pfizer"), "Pfizer")

    def test_get_sponsor_variations(self):
        """Test that getting variations works for aliases and canonical names."""
        # Test with alias
        vars_jnj = get_sponsor_variations("J&J")
        self.assertIn("Janssen Research & Development, LLC", vars_jnj)
        self.assertIn("Janssen", vars_jnj)
        
        # Test with canonical
        vars_janssen = get_sponsor_variations("Janssen")
        self.assertEqual(vars_jnj, vars_janssen)
        
        # Test with variation input (should normalize first)
        vars_variation = get_sponsor_variations("Janssen Research & Development, LLC")
        self.assertEqual(vars_janssen, vars_variation)

    def test_unknown_sponsor(self):
        """Test behavior for unknown sponsors."""
        self.assertEqual(normalize_sponsor("Unknown Pharma"), "Unknown Pharma")
        self.assertIsNone(get_sponsor_variations("Unknown Pharma"))

if __name__ == "__main__":
    unittest.main()
