import unittest

from oracle_model import TransformerOracle, build_oracle


class OracleModelTests(unittest.TestCase):
    def test_transformer_oracle_respects_num_blocks(self):
        oracle = build_oracle(
            oracle_type="transformer",
            in_dim=768,
            small_dim=128,
            num_blocks=2,
        )

        self.assertIsInstance(oracle, TransformerOracle)
        self.assertEqual(len(oracle.blocks), 2)


if __name__ == "__main__":
    unittest.main()
