from dataclasses import dataclass
from typing import Literal, Dict


Polarity = Literal["positive", "negative", "absolute"]


@dataclass(frozen=True)
class ERPWindow:
    name: str
    tmin: float
    tmax: float
    polarity: Polarity = "absolute"


# Canonical named windows
ERP_WINDOWS: Dict[str, ERPWindow] = {
    "MMN": ERPWindow(
        name="MMN",
        tmin=0.15,
        tmax=0.25,
        polarity="negative",
    ),
    "P300": ERPWindow(
        name="P300",
        tmin=0.30,
        tmax=0.50,
        polarity="positive",
    ),
}