import unittest
import json
import logging
from unittest.mock import MagicMock, patch
import dspy

from src.llm.dspy_pipeline import DSPySchemeExtractor
from src.config import ExtractionConfig
from src.models import SchemeHeader

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

class TestDSPyExtraction(unittest.TestCase):
    def setUp(self):
        self.config = ExtractionConfig(
            openrouter_api_key="test_key",
            openrouter_model="test_model"
        )
        self.mock_llm = MagicMock(spec=dspy.LM)
        self.mock_llm.model_name = "test_model"
        
        # Initialize extractor
        self.extractor = DSPySchemeExtractor(self.mock_llm, self.config)

    def test_successful_extraction(self):
        """Test successful extraction of a valid scheme."""
        # Mock LLM response
        mock_json = {
            "schemes": [{
                "scheme_name": "Test Scheme",
                "scheme_type": "BUY_SIDE",
                "scheme_subtype": "PERIODIC_CLAIM",
                "duration": "01/01/2025 to 31/03/2025",
                "max_cap": 50000,
                "fsn_file_config_file": "Yes"
            }]
        }
        
        # Mock the dspy.Predict module call
        mock_prediction = MagicMock()
        mock_prediction.schemes_json = json.dumps(mock_json)
        self.extractor.extract_module = MagicMock(return_value=mock_prediction)
        
        # Run extraction
        response = self.extractor.extract("Test Subject", "Test Body")
        
        # Verify
        self.assertEqual(len(response.schemes), 1)
        scheme = response.schemes[0]
        self.assertEqual(scheme.scheme_name, "Test Scheme")
        self.assertEqual(scheme.scheme_type, "BUY_SIDE")
        self.assertEqual(scheme.max_cap, "50000")  # Should be converted to string
        self.assertEqual(scheme.fsn_file_config_file, "Yes")

    def test_list_response_handling(self):
        """Test handling of direct list response from LLM."""
        mock_list = [{
            "scheme_name": "List Scheme",
            "scheme_type": "ONE_OFF",
            "scheme_subtype": "ONE_OFF"
        }]
        
        mock_prediction = MagicMock()
        mock_prediction.schemes_json = json.dumps(mock_list)
        self.extractor.extract_module = MagicMock(return_value=mock_prediction)
        
        response = self.extractor.extract("Subject", "Body")
        
        self.assertEqual(len(response.schemes), 1)
        self.assertEqual(response.schemes[0].scheme_name, "List Scheme")

    def test_single_object_response_handling(self):
        """Test handling of single object response from LLM."""
        mock_obj = {
            "scheme_name": "Single Scheme",
            "scheme_type": "SELL_SIDE",
            "scheme_subtype": "COUPON"
        }
        
        mock_prediction = MagicMock()
        mock_prediction.schemes_json = json.dumps(mock_obj)
        self.extractor.extract_module = MagicMock(return_value=mock_prediction)
        
        response = self.extractor.extract("Subject", "Body")
        
        self.assertEqual(len(response.schemes), 1)
        self.assertEqual(response.schemes[0].scheme_name, "Single Scheme")

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        mock_prediction = MagicMock()
        mock_prediction.schemes_json = "Invalid JSON"
        self.extractor.extract_module = MagicMock(return_value=mock_prediction)
        
        response = self.extractor.extract("Subject", "Body")
        
        self.assertEqual(len(response.schemes), 0)
        self.assertIn("Invalid JSON", response.raw_response)

if __name__ == '__main__':
    unittest.main()
