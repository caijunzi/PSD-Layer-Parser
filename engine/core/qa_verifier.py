import os
from typing import Dict, Any
from psd_tools import PSDImage
from psd_tools.constants import Resource

class QaVerifier:
    """Rigorous pre-flight QA auditor for print-ready PSD/PSB files using psd-tools."""

    @staticmethod
    def audit_psd(file_path: str, expected_dpi: float = 150.0, expected_width_mm: float = 900.0, expected_height_mm: float = 1600.0) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target PSD does not exist: {file_path}")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        psd = PSDImage.open(file_path)

        res_info = psd.image_resources.get_data(Resource.RESOLUTION_INFO)
        dpi_h = res_info.horizontal / 65536.0 if res_info else 72.0
        dpi_v = res_info.vertical / 65536.0 if res_info else 72.0

        phys_w_mm = psd.width / dpi_h * 25.4
        phys_h_mm = psd.height / dpi_v * 25.4

        layers_info = []
        for i, lyr in enumerate(psd):
            clean_name = lyr.name.replace('\x00', '').strip()
            layers_info.append({
                'index': i + 1,
                'name': clean_name,
                'width': lyr.width,
                'height': lyr.height,
                'visible': lyr.is_visible()
            })

        # Verification asserts
        dpi_match = abs(dpi_h - expected_dpi) < 0.1 and abs(dpi_v - expected_dpi) < 0.1
        w_mm_match = abs(phys_w_mm - expected_width_mm) < 1.0
        h_mm_match = abs(phys_h_mm - expected_height_mm) < 1.0
        cmyk_mode = psd.color_mode == 4  # 4 is CMYK

        passed = dpi_match and w_mm_match and h_mm_match and cmyk_mode and len(layers_info) > 0

        report = {
            'file_path': file_path,
            'file_size_mb': round(file_size_mb, 2),
            'pixel_width': psd.width,
            'pixel_height': psd.height,
            'dpi_h': round(dpi_h, 1),
            'dpi_v': round(dpi_v, 1),
            'physical_width_mm': round(phys_w_mm, 1),
            'physical_height_mm': round(phys_h_mm, 1),
            'color_mode': str(psd.color_mode),
            'layer_count': len(layers_info),
            'layers': layers_info,
            'passed': passed
        }
        return report
