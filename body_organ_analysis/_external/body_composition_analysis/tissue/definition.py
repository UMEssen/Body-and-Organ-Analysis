import enum

from body_composition_analysis.body_regions.definition import BodyRegion


class HURange(enum.Enum):
    ALL = (-1000, 3000)
    ADIPOSE_TISSUE = (-190, -30)
    MUSCLE_TISSUE = (-29, 150)


class Tissue(enum.IntEnum):
    MUSCLE = 1
    BONE = 2
    SAT = 3
    VAT = 4
    IMAT = 5
    PAT = 6
    EAT = 7


TISSUE_DERIVATION_RULES = {
    Tissue.MUSCLE: (HURange.MUSCLE_TISSUE, BodyRegion.MUSCLE),
    Tissue.BONE: (HURange.ALL, BodyRegion.BONE),
    Tissue.SAT: (HURange.ADIPOSE_TISSUE, BodyRegion.SUBCUTANEOUS_TISSUE),
    Tissue.VAT: (HURange.ADIPOSE_TISSUE, BodyRegion.ABDOMINAL_CAVITY),
    Tissue.IMAT: (HURange.ADIPOSE_TISSUE, BodyRegion.MUSCLE),
    Tissue.PAT: (HURange.ADIPOSE_TISSUE, BodyRegion.MEDIASTINUM),
    Tissue.EAT: (HURange.ADIPOSE_TISSUE, BodyRegion.PERICARDIUM),
}
