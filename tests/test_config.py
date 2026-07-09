import unittest

from config import build_experiment_config, get_config


class ExperimentConfigTests(unittest.TestCase):
    def test_default_distilgpt2_paths_are_backward_compatible(self):
        cfg = get_config("distilgpt2")

        self.assertEqual(cfg.data_dir, "oracle_data_distilgpt2")
        self.assertEqual(cfg.checkpoint_path, "oracle_distilgpt2.pth")
        self.assertEqual(cfg.small_dim, 96)
        self.assertEqual(cfg.cos_weight, 0.1)
        self.assertEqual(cfg.num_blocks, 1)

    def test_run_tag_derives_isolated_artifact_paths_and_overrides(self):
        cfg = build_experiment_config(
            preset="distilgpt2",
            samples=1024,
            max_length=128,
            oracle_type="transformer",
            small_dim=128,
            cos_weight=0.5,
            num_blocks=1,
            batch_size=2,
            epochs=3,
            run_tag="distilgpt2_wiki1024_len128_tr_sdim128_cos0.5_e3",
        )

        self.assertEqual(cfg.collect_samples, 1024)
        self.assertEqual(cfg.max_length, 128)
        self.assertEqual(cfg.oracle_type, "transformer")
        self.assertEqual(cfg.small_dim, 128)
        self.assertEqual(cfg.cos_weight, 0.5)
        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.num_epochs, 3)
        self.assertEqual(
            cfg.data_dir,
            "oracle_data_distilgpt2_wiki1024_len128_tr_sdim128_cos0.5_e3",
        )
        self.assertEqual(
            cfg.checkpoint_path,
            "oracle_distilgpt2_wiki1024_len128_tr_sdim128_cos0.5_e3.pth",
        )


if __name__ == "__main__":
    unittest.main()
