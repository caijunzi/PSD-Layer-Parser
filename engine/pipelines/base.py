from abc import ABC, abstractmethod
from typing import Dict, Any
from engine.core.models import ProcessingContext
from engine.core.psd_compiler import PsdCompiler
from engine.core.qa_verifier import QaVerifier

class BasePipeline(ABC):
    """Abstract base class defining standard lifecycle for domain-specific layer separation pipelines."""

    def __init__(self, name: str):
        self.name = name

    def execute(self, context: ProcessingContext, output_path: str) -> Dict[str, Any]:
        """
        Executes standard pipeline lifecycle:
        1. Pre-process (geometry & lighting)
        2. Extract features & separate masks
        3. Build layer stack
        4. Compile PSD/PSB
        5. Verify with automated QA auditor
        """
        self.prepare(context)
        self.extract_features(context)
        self.build_layers(context)
        
        compiled_path = PsdCompiler.compile_psd(context, output_path)
        qa_report = QaVerifier.audit_psd(
            compiled_path,
            expected_dpi=context.target_dpi,
            expected_width_mm=context.physical_width_mm,
            expected_height_mm=context.physical_height_mm
        )
        context.metadata['qa_report'] = qa_report
        return qa_report

    def prepare(self, context: ProcessingContext):
        """Default preparation step. Overridden by pipelines as needed."""
        if context.rectified_image is None:
            context.rectified_image = context.raw_image.copy()

    @abstractmethod
    def extract_features(self, context: ProcessingContext):
        """Extract domain-specific masks (e.g. micro-holes, foil, ink stages)."""
        pass

    @abstractmethod
    def build_layers(self, context: ProcessingContext):
        """Assemble LayerDescriptor instances into context.layers."""
        pass
