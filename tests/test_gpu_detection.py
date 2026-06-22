# Tests for GPU detection and capability checking
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import cooler_verdict as temp_compare


def test_detect_nvidia_gpu_found():
    """Test NVIDIA GPU detection when nvidia-smi is available."""
    with patch('cooler_verdict.shutil.which', return_value='/usr/bin/nvidia-smi'):
        with patch('cooler_verdict.subprocess.check_output', return_value="NVIDIA RTX 3080\n"):
            result = temp_compare.detect_nvidia_gpu()
    assert result == "NVIDIA RTX 3080"


def test_detect_nvidia_gpu_not_found():
    """Test NVIDIA GPU detection when nvidia-smi is not available."""
    with patch('cooler_verdict.shutil.which', return_value=None):
        result = temp_compare.detect_nvidia_gpu()
    assert result is None


def test_detect_nvidia_gpu_empty_output():
    """Test NVIDIA GPU detection with empty output."""
    with patch('cooler_verdict.shutil.which', return_value='/usr/bin/nvidia-smi'):
        with patch('cooler_verdict.subprocess.check_output', return_value=""):
            result = temp_compare.detect_nvidia_gpu()
    assert result is None


def test_detect_amd_gpu_via_rocm_smi():
    """Test AMD GPU detection via rocm-smi."""
    with patch('cooler_verdict.shutil.which') as mock_which:
        mock_which.side_effect = lambda cmd: '/usr/bin/rocm-smi' if cmd == 'rocm-smi' else None
        with patch('cooler_verdict.subprocess.check_output', return_value="GPU ID\ngpu0 gfx90a\n"):
            result = temp_compare.detect_amd_gpu()
    assert result == "AMD Radeon (ROCm)"


def test_detect_amd_gpu_not_found():
    """Test AMD GPU detection when no AMD GPU tools are available."""
    with patch('cooler_verdict.shutil.which', return_value=None):
        with patch('cooler_verdict.Path.exists', return_value=False):
            result = temp_compare.detect_amd_gpu()
    assert result is None


def test_detect_intel_gpu_via_lspci():
    """Test Intel GPU detection via lspci."""
    lspci_output = """
00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 630
01:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti]
"""
    with patch('cooler_verdict.shutil.which') as mock_which:
        mock_which.side_effect = lambda cmd: '/usr/bin/lspci' if cmd == 'lspci' else None
        with patch('cooler_verdict.subprocess.check_output', return_value=lspci_output):
            result = temp_compare.detect_intel_gpu()
    assert result is not None
    assert "Intel" in result or "UHD" in result


def test_detect_intel_gpu_via_i915_module():
    """Test Intel GPU detection via i915 kernel module."""
    with patch('cooler_verdict.shutil.which', return_value=None):
        with patch('cooler_verdict.shutil.which', return_value=None):
            # Create a mock Path class that tracks what paths are being checked
            mock_path_instances = {}
            
            class MockPath:
                def __init__(self, path):
                    self.path = str(path)
                    mock_path_instances[self.path] = self
                
                def exists(self):
                    return self.path == "/sys/module/i915"
            
            with patch('cooler_verdict.Path', MockPath):
                result = temp_compare.detect_intel_gpu()
    assert result == "Intel GPU (i915)"


def test_get_gpu_info_nvidia_available():
    """Test get_gpu_info when NVIDIA GPU is available and working."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value="NVIDIA RTX 3080"):
        with patch('cooler_verdict.check_gpu_can_run_code', return_value=True):
            info = temp_compare.get_gpu_info()
    assert info["vendor"] == "nvidia"
    assert info["device"] == "NVIDIA RTX 3080"
    assert info["available"] is True
    assert info["can_run_code"] is True
    assert "NVIDIA" in info["message"]


def test_get_gpu_info_amd_available():
    """Test get_gpu_info when AMD GPU is available."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value=None):
        with patch('cooler_verdict.detect_amd_gpu', return_value="AMD Radeon (ROCm)"):
            with patch('cooler_verdict.check_gpu_can_run_code', return_value=True):
                info = temp_compare.get_gpu_info()
    assert info["vendor"] == "amd"
    assert info["available"] is True
    assert info["can_run_code"] is True


def test_get_gpu_info_intel_available():
    """Test get_gpu_info when Intel GPU is available."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value=None):
        with patch('cooler_verdict.detect_amd_gpu', return_value=None):
            with patch('cooler_verdict.detect_intel_gpu', return_value="Intel GPU (i915)"):
                with patch('cooler_verdict.check_gpu_can_run_code', return_value=True):
                    info = temp_compare.get_gpu_info()
    assert info["vendor"] == "intel"
    assert info["available"] is True


def test_get_gpu_info_no_gpu():
    """Test get_gpu_info when no GPU is detected."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value=None):
        with patch('cooler_verdict.detect_amd_gpu', return_value=None):
            with patch('cooler_verdict.detect_intel_gpu', return_value=None):
                info = temp_compare.get_gpu_info()
    assert info["vendor"] == "none"
    assert info["available"] is False
    assert info["can_run_code"] is False


def test_get_gpu_info_detected_but_not_runnable():
    """Test get_gpu_info when GPU is detected but cannot run code."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value="NVIDIA RTX 3080"):
        with patch('cooler_verdict.check_gpu_can_run_code', return_value=False):
            info = temp_compare.get_gpu_info()
    assert info["vendor"] == "nvidia"
    assert info["available"] is True
    assert info["can_run_code"] is False


def test_check_gpu_can_run_code_nvidia():
    """Test GPU capability check for NVIDIA."""
    with patch('cooler_verdict.subprocess.check_output', return_value="GPU 0: NVIDIA RTX 3080\n"):
        result = temp_compare.check_gpu_can_run_code("nvidia")
    assert result is True


def test_check_gpu_can_run_code_nvidia_failed():
    """Test GPU capability check for NVIDIA when check fails."""
    with patch('cooler_verdict.subprocess.check_output', side_effect=Exception("nvidia-smi failed")):
        result = temp_compare.check_gpu_can_run_code("nvidia")
    assert result is False


def test_check_gpu_can_run_code_amd_with_rocm():
    """Test GPU capability check for AMD with ROCm."""
    with patch('cooler_verdict.shutil.which', return_value='/usr/bin/rocm-smi'):
        with patch('cooler_verdict.subprocess.check_output', return_value="GPU ID\ngpu0\n"):
            result = temp_compare.check_gpu_can_run_code("amd")
    assert result is True


def test_check_gpu_can_run_code_amd_without_rocm():
    """Test GPU capability check for AMD without ROCm installed."""
    with patch('cooler_verdict.shutil.which', return_value=None):
        result = temp_compare.check_gpu_can_run_code("amd")
    assert result is False


def test_check_gpu_can_run_code_intel():
    """Test GPU capability check for Intel GPU."""
    with patch('cooler_verdict.Path.exists') as mock_exists:
        mock_exists.return_value = True
        result = temp_compare.check_gpu_can_run_code("intel")
    assert result is True


def test_check_gpu_can_run_code_intel_not_available():
    """Test GPU capability check for Intel GPU when driver is not loaded."""
    with patch('cooler_verdict.Path.exists', return_value=False):
        result = temp_compare.check_gpu_can_run_code("intel")
    assert result is False


def test_check_gpu_can_run_code_unknown():
    """Test GPU capability check for unknown GPU vendor."""
    result = temp_compare.check_gpu_can_run_code("unknown")
    assert result is False


def test_infer_gpu_vendor_returns_correct_vendor():
    """Test that infer_gpu_vendor returns vendor from get_gpu_info."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value="NVIDIA RTX 3080"):
        with patch('cooler_verdict.check_gpu_can_run_code', return_value=True):
            vendor = temp_compare.infer_gpu_vendor()
    assert vendor == "nvidia"


def test_get_gpu_stress_cmd_nvidia():
    """Test GPU stress command generation for NVIDIA."""
    with patch('cooler_verdict.get_gpu_info') as mock_info:
        mock_info.return_value = {
            "vendor": "nvidia",
            "device": "NVIDIA RTX 3080",
            "available": True,
            "can_run_code": True,
        }
        with patch('cooler_verdict.shutil.which', return_value='/usr/bin/nvidia-smi'):
            cmd = temp_compare.get_gpu_stress_cmd()
    assert cmd is not None
    assert "nvidia-smi" in cmd


def test_get_gpu_stress_cmd_amd():
    """Test GPU stress command generation for AMD."""
    with patch('cooler_verdict.get_gpu_info') as mock_info:
        mock_info.return_value = {
            "vendor": "amd",
            "device": "AMD Radeon (ROCm)",
            "available": True,
            "can_run_code": True,
        }
        with patch('cooler_verdict.shutil.which', return_value='/usr/bin/rocm-smi'):
            cmd = temp_compare.get_gpu_stress_cmd()
    assert cmd is not None
    assert "rocm-smi" in cmd


def test_get_gpu_stress_cmd_intel():
    """Test GPU stress command generation for Intel."""
    with patch('cooler_verdict.get_gpu_info') as mock_info:
        mock_info.return_value = {
            "vendor": "intel",
            "device": "Intel GPU (i915)",
            "available": True,
            "can_run_code": True,
        }
        with patch('cooler_verdict.shutil.which', return_value='/usr/bin/intel_gpu_top'):
            cmd = temp_compare.get_gpu_stress_cmd()
    assert cmd is not None


def test_get_gpu_stress_cmd_no_gpu():
    """Test GPU stress command generation when no GPU is available."""
    with patch('cooler_verdict.get_gpu_info') as mock_info:
        mock_info.return_value = {
            "vendor": "none",
            "device": None,
            "available": False,
            "can_run_code": False,
        }
        cmd = temp_compare.get_gpu_stress_cmd()
    assert cmd is None


def test_start_stressors_includes_auto_gpu_stress():
    """Test that start_stressors includes auto-generated GPU stress."""
    with patch('cooler_verdict.shutil.which') as mock_which:
        mock_which.side_effect = lambda cmd: '/usr/bin/stress-ng' if cmd == 'stress-ng' else None
        with patch('cooler_verdict.get_gpu_stress_cmd', return_value="while true; do nvidia-smi > /dev/null; sleep 1; done"):
            with patch('cooler_verdict.run_subprocess') as mock_run:
                mock_run.return_value = MagicMock()
                stressors = temp_compare.start_stressors(cpu_workers=4, gpu_stress_cmd=None)
    
    # Should have both CPU and GPU stressors
    assert mock_run.call_count == 2
    # Check that gpu-stress-auto was passed as the name parameter (second argument)
    stress_names = [call[0][1] for call in mock_run.call_args_list]
    assert "gpu-stress-auto" in stress_names


def test_start_stressors_uses_custom_gpu_stress():
    """Test that start_stressors uses custom GPU stress when provided."""
    custom_cmd = "my-custom-gpu-stress"
    with patch('cooler_verdict.shutil.which') as mock_which:
        mock_which.side_effect = lambda cmd: '/usr/bin/stress-ng' if cmd == 'stress-ng' else None
        with patch('cooler_verdict.run_subprocess') as mock_run:
            mock_run.return_value = MagicMock()
            stressors = temp_compare.start_stressors(cpu_workers=4, gpu_stress_cmd=custom_cmd)
    
    # Should have both CPU and GPU stressors
    assert mock_run.call_count == 2
    # Check that custom command was used
    calls = [str(call) for call in mock_run.call_args_list]
    assert any(custom_cmd in call for call in calls)


def test_gpu_detection_graceful_error_handling():
    """Test that GPU detection handles errors gracefully."""
    # Simulate various error conditions
    with patch('cooler_verdict.subprocess.check_output', side_effect=Exception("Command failed")):
        # Should not raise, just return None/empty results
        result = temp_compare.detect_nvidia_gpu()
    assert result is None


def test_gpu_info_message_field():
    """Test that GPU info contains informative message."""
    with patch('cooler_verdict.detect_nvidia_gpu', return_value="NVIDIA RTX 3080"):
        with patch('cooler_verdict.check_gpu_can_run_code', return_value=True):
            info = temp_compare.get_gpu_info()
    assert "message" in info
    assert len(info["message"]) > 0
    assert "NVIDIA" in info["message"] or "detected" in info["message"].lower()
