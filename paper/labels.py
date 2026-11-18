"""Public labels: participants P1-P4 and sessions Pk-j (j = chronological order within participant)."""
PART = {"P01": "P1", "P02": "P2", "P03": "P3", "P09": "P4"}
SESSIONS = ['P01_2026-09-26_09-28-01', 'P01_2026-09-26_10-30-28', 'P01_2026-09-26_17-15-09',
            'sesion_2026-09-18_09-57-46', 'sesion_2026-09-18_13-22-02',  # legacy-format sessions of P2
            'P02_2026-09-26_08-35-05', 'P02_2026-09-26_09-27-48', 'P02_2026-09-28_08-21-45', 'P02_2026-09-28_13-24-09', 'P02_2026-09-30_07-55-50',
            'P03_2026-09-26_08-29-43', 'P03_2026-09-26_09-41-23', 'P03_2026-09-27_04-03-44', 'P03_2026-09-28_08-45-50',
            'P09_2026-09-30_16-02-43']


def owner(sid):
    return PART["P02" if sid.startswith("sesion_") else sid[:3]]


def _key(sid):
    return sid.split("_", 1)[1]  # date_time, chronological


def label(sid, star=True):
    p = owner(sid); mine = sorted([s for s in SESSIONS if owner(s) == p], key=_key)
    return f"{p}-{mine.index(sid) + 1}" + ("*" if star and sid.startswith("sesion_") else "")


def is_legacy(sid):
    return sid.startswith("sesion_")
