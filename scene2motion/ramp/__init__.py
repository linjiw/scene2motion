"""Response-adaptive motion-program components.

RAMP deliberately lives outside :mod:`scene2motion.program`: the latter is the frozen 43-D
audit representation used by historical checkpoints and ledgers.  New event-aligned packet
representations can evolve here without changing those artifacts' meaning.
"""

from .packet import (
    CoherentPacketPair,
    CoherentMotionPacket,
    MAX_PACKET_STRENGTH,
    PACKET_SCHEMA_VERSION,
    PacketControls,
    PacketRenderInfo,
    PhaseMatch,
    SO3_BRANCH_MARGIN_RAD,
    constraint_support_digest,
    extract_absolute_packet,
    extract_packet_pair,
    extract_residual_packet,
    render_packet,
)
from .phase import (
    PHASE_ALIGNMENT_METHOD,
    TARGET_PHASE_ALIGNMENT_METHOD,
    StanceEvidence,
    TargetPhaseMatch,
    align_cyclic_phase_windows,
    align_target_phase_window,
)
from .step_phase import (
    STEP_PHASE_METHOD,
    STEP_PHASE_PROTOCOL_VERSION,
    StepPhaseCycle,
    StepPhaseLandmarks,
    align_step_phase_cycles,
    align_step_target_phase,
    enumerate_step_phase_cycles,
    enumerate_step_phase_cycles_from_qpos,
    step_phase_cycle_from_qpos,
    step_phase_measurement_protocol_hash,
    validate_step_phase_cycle,
)

__all__ = [
    "CoherentPacketPair",
    "CoherentMotionPacket",
    "MAX_PACKET_STRENGTH",
    "PACKET_SCHEMA_VERSION",
    "PacketControls",
    "PacketRenderInfo",
    "PHASE_ALIGNMENT_METHOD",
    "TARGET_PHASE_ALIGNMENT_METHOD",
    "PhaseMatch",
    "SO3_BRANCH_MARGIN_RAD",
    "STEP_PHASE_METHOD",
    "STEP_PHASE_PROTOCOL_VERSION",
    "StepPhaseCycle",
    "StepPhaseLandmarks",
    "constraint_support_digest",
    "StanceEvidence",
    "TargetPhaseMatch",
    "align_cyclic_phase_windows",
    "align_step_phase_cycles",
    "align_step_target_phase",
    "align_target_phase_window",
    "enumerate_step_phase_cycles",
    "enumerate_step_phase_cycles_from_qpos",
    "extract_absolute_packet",
    "extract_packet_pair",
    "extract_residual_packet",
    "render_packet",
    "step_phase_cycle_from_qpos",
    "step_phase_measurement_protocol_hash",
    "validate_step_phase_cycle",
]
