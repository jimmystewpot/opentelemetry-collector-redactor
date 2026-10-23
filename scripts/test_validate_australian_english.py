import unittest
from validate_australian_english import check_text, scan_file

class TestValidateAustralianEnglish(unittest.TestCase):
    def test_australian_english_passes(self):
        au_text = """
        # Architecture Overview
        We optimise the quantised transformer model artefacts with specialised behaviour.
        All telemetry pipelines initialise cleanly under the Apache Licence, Version 2.0.
        Centralised logging centres on low latency and serialisation efficiency.
        """
        violations = check_text(au_text)
        self.assertEqual(violations, [], f"Expected 0 violations, got {violations}")

    def test_american_english_fails(self):
        us_text = """
        # Architecture Overview
        We optimize the quantized model artifacts with specialized behavior.
        The system will initialize under Apache License.
        Centralized logging centers on serialization efficiency.
        The optimizer and quantizer assist the serializer and initializer.
        He was centering the text while it was centered.
        """
        violations = check_text(us_text)
        expected_words = {
            "optimize", "quantized", "artifacts", "specialized", "behavior",
            "initialize", "license", "centralized", "centers", "serialization",
            "optimizer", "quantizer", "serializer", "initializer",
            "centering", "centered"
        }
        detected_words = {v["matched"].lower() for v in violations}
        self.assertTrue(expected_words.issubset(detected_words), f"Missing violations: {expected_words - detected_words}")

    def test_code_blocks_ignored(self):
        text_with_code = """
        Here is an Australian sentence where we optimise code.
        ```python
        def optimize_model():
            # In code blocks, library names like optimize or artifact are allowed
            return "artifact"
        ```
        And inline code like `onnx.optimize()` or `artifact.tar.gz` should be ignored.
        """
        violations = check_text(text_with_code)
        self.assertEqual(violations, [], f"Expected 0 violations in code blocks, got {violations}")

if __name__ == "__main__":
    unittest.main()
