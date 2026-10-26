
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import answer


class TestAnswerPipeline(unittest.TestCase):

    def setUp(self):
        self.passages = [
            {
                "chunk_id": 1,
                "text": (
                    "Natural language processing enables "
                    "computers to process human language."
                ),
                "source_pages": [8],
            },
            {
                "chunk_id": 2,
                "text": (
                    "Text classification assigns text "
                    "to predefined categories."
                ),
                "source_pages": [9, 10],
            },
        ]

    def test_context_includes_source_pages_and_text(self):
        context = answer.build_context(self.passages)

        self.assertIn("[Page 8]", context)
        self.assertIn("[Page 9]", context)
        self.assertIn("[Page 10]", context)
        self.assertIn("Natural language processing", context)

    def test_valid_citations_are_accepted(self):
        response = (
            "NLP processes human language [Page 8]. "
            "Text classification assigns categories [Page 9]."
        )

        warnings = answer.validate_citations(
            response, self.passages
        )

        self.assertEqual(warnings, [])

    def test_unsupported_page_is_flagged(self):
        response = "This claim comes from the notes [Page 99]."

        warnings = answer.validate_citations(
            response, self.passages
        )

        self.assertTrue(
            any("99" in warning for warning in warnings)
        )

    def test_missing_citations_are_flagged(self):
        response = "NLP helps computers process language."

        warnings = answer.validate_citations(
            response, self.passages
        )

        self.assertTrue(
            any(
                "No [Page N] citations" in warning
                for warning in warnings
            )
        )

    def test_generation_works_with_mock_response(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = (
            SimpleNamespace(
                output_text=(
                    "NLP processes human language [Page 8]."
                )
            )
        )

        with patch("builtins.print"):
            result = answer.generate_answer(
                mock_client,
                "What is NLP?",
                self.passages,
            )

        self.assertEqual(
            result,
            "NLP processes human language [Page 8].",
        )
        mock_client.interactions.create.assert_called_once()

    def test_generation_prints_unsupported_citation_warning(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = (
            SimpleNamespace(
                output_text="NLP is useful [Page 99]."
            )
        )

        with patch("builtins.print") as mock_print:
            result = answer.generate_answer(
                mock_client,
                "What is NLP?",
                self.passages,
            )

        self.assertEqual(
            result,
            "NLP is useful [Page 99].",
        )

        printed_output = " ".join(
            str(call) for call in mock_print.call_args_list
        )

        self.assertIn("Citation Warnings", printed_output)
        self.assertIn("99", printed_output)

    def test_empty_generation_response_is_rejected(self):
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = (
            SimpleNamespace(output_text="")
        )

        with patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                answer.generate_answer(
                    mock_client,
                    "What is NLP?",
                    self.passages,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
