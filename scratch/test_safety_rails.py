import sys
import os
import unittest
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.safety_rails import run_pre_inference_validation, SafetyRailException

class TestSafetyRails(unittest.TestCase):
    def test_too_small_file(self):
        # File smaller than 132 bytes should fail
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(b"small", "test.dcm")
        self.assertIn("too small to be a valid DICOM", str(ctx.exception))

    def test_missing_dicom_magic_bytes(self):
        # File of 132 bytes but missing 'DICM' at offset 128
        content = b"a" * 128 + b"NOTD"
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(content, "test.dcm")
        self.assertIn("Missing DICOM magic bytes 'DICM'", str(ctx.exception))

    def test_corrupted_dicom_header(self):
        # File has DICM magic bytes but invalid header fields
        content = b"a" * 128 + b"DICM" + b"randomjunkhere"
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(content, "test.dcm")
        self.assertTrue(
            "DICOM file corruption detected" in str(ctx.exception) or
            "Missing essential DICOM tag" in str(ctx.exception)
        )

    def test_invalid_nifti_size(self):
        # File smaller than 348 bytes
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(b"small", "test.nii")
        self.assertIn("too small to be a valid NIfTI", str(ctx.exception))

    def test_invalid_nifti_header_size(self):
        # 348 bytes but incorrect header size magic in first 4 bytes
        content = b"\x00\x00\x00\x00" + b"a" * 344
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(content, "test.nii")
        self.assertIn("Invalid NIfTI header structure", str(ctx.exception))

    def test_invalid_image_format(self):
        # Non-image content with .png extension
        content = b"not an image file content"
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(content, "test.png")
        self.assertIn("Image file corruption or unsupported", str(ctx.exception))

    def test_unsupported_extension(self):
        # Unsupported extension should fail
        with self.assertRaises(SafetyRailException) as ctx:
            run_pre_inference_validation(b"dummy", "test.txt")
        self.assertIn("Unsupported clinical imaging format", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
