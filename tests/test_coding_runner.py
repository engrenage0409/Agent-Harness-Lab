import pytest
from coding.test_runner import run_tests

def test_approval_required(tmp_path):
    with pytest.raises(PermissionError):
        run_tests(tmp_path,approved=False)
