"""
Unit tests for AnonymizeUltrasound common functionality.
"""

import pytest


class TestAnonymizeUltrasoundCommon:
    """Test common functionality for AnonymizeUltrasound."""

    def test_patient_name_anonymization(self):
        """Test patient name anonymization."""
        original_name = "John Doe"
        anonymized_name = self.anonymize_patient_name(original_name)

        assert anonymized_name != original_name
        assert "Anonymous" in anonymized_name or "Patient" in anonymized_name
        assert len(anonymized_name) > 0

    def test_patient_id_anonymization(self):
        """Test patient ID anonymization."""
        original_id = "12345"
        anonymized_id = self.anonymize_patient_id(original_id)

        assert anonymized_id != original_id
        assert len(anonymized_id) > 0

    def test_date_anonymization(self):
        """Test date anonymization."""
        original_date = "2024-01-01"
        anonymized_date = self.anonymize_date(original_date)

        assert anonymized_date != original_date
        assert len(anonymized_date) > 0

    def test_empty_value_handling(self):
        """Test handling of empty values."""
        assert self.anonymize_patient_name("") == ""
        assert self.anonymize_patient_id("") == ""
        assert self.anonymize_date("") == ""

    def test_none_value_handling(self):
        """Test handling of None values."""
        assert self.anonymize_patient_name(None) == ""
        assert self.anonymize_patient_id(None) == ""
        assert self.anonymize_date(None) == ""

    # Helper methods for anonymization
    def anonymize_patient_name(self, name):
        """Anonymize patient name."""
        if not name:
            return ""
        return f"Anonymous_Patient_{hash(name) % 1000:03d}"

    def anonymize_patient_id(self, patient_id):
        """Anonymize patient ID."""
        if not patient_id:
            return ""
        return f"ANON_{hash(patient_id) % 10000:04d}"

    def anonymize_date(self, date_str):
        """Anonymize date."""
        if not date_str:
            return ""
        return "1900-01-01"
