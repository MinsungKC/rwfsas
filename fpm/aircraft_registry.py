"""ICAO24 -> known-firefighting-aircraft registry (rebuild-spec Section 8 item 2).

"No aircraft-type knowledge... Classification logic that doesn't know a
Type-1 helo from an LAT will never work." This resolves that, keyed
directly by ICAO24 hex (what every ADS-B state vector already carries), so
no FAA N-Number <-> Mode-S cross-reference step is needed.

Data source, and why: the FAA's own ReleasableAircraft.zip
(registry.faa.gov) returned HTTP 403/503 from Akamai bot protection when
probed from this environment on 2026-09-17 -- a documented access barrier,
not a design choice. OpenSky Network publishes a free, unauthenticated
global aircraft metadata CSV keyed by icao24 (~94 MB,
https://opensky-network.org/aircraft-database) that covers the same ground
for this purpose and is directly joinable to live ADS-B traffic.

Precision over recall, deliberately: an aircraft only enters the fleet
index when its OPERATOR/OWNER field matches a known wildland-fire aviation
operator (10 Tanker, Coulson, Neptune, CAL FIRE, USFS, ...). Matching on
MODEL alone (e.g. "any Bell 206 or King Air is a tanker") would flood on
ordinary GA/tour/EMS traffic -- exactly the false-positive class rebuild-spec
item 5 warns against. Once an aircraft is confirmed-fire by operator, its
model determines a ROLE (vlat/lat/leadplane/helitanker_typeN) via a separate
table below.

FIRE_OPERATORS and MODEL_ROLE are necessarily incomplete curated lists, not
an exhaustive fleet -- extend them as specific tail numbers/operators are
identified. A miss here falls back to aircraft_fire_edge's callsign-prefix
and loitering-behavior heuristics, it does not block tracking.
"""
import csv
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '_cache')
RAW_CSV = os.path.join(CACHE_DIR, 'opensky_aircraft_db.csv')
FLEET_JSON = os.path.join(CACHE_DIR, 'firefighting_fleet.json')
RAW_URL = 'https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv'
RAW_MAX_AGE_DAYS = 30  # aircraft ownership/registration changes slowly

FIRE_OPERATORS = [
    '10 TANKER', 'TEN TANKER', 'COULSON', 'NEPTUNE AVIATION', 'AERO-FLITE',
    'AERO FLITE', 'AEROFLITE', 'CAL FIRE', 'CALFIRE', 'CAL-FIRE',
    'CALIFORNIA DEPARTMENT OF FORESTRY', 'CDF AVIATION',
    'US FOREST SERVICE', 'USDA FOREST SERVICE', 'FOREST SERVICE',
    'BUREAU OF LAND MANAGEMENT', 'BRIDGER AEROSPACE', 'ERICKSON INC',
    'ERICKSON AIR', 'ERICKSON AERO', 'COLUMBIA HELICOPTERS',
    'GLOBAL SUPERTANKER', 'DAUNTLESS AIR', 'KACHINA AVIATION',
    'ROGERS HELICOPTER', 'BUTLER AIRCRAFT', 'MINDEN AIR', 'CONAIR GROUP',
    'CONAIR AVIATION', 'ABSOLUTE AVIATION', 'INTERMOUNTAIN HELICOPTER',
    'SIS-Q FLYING', 'SISQ', 'HELICOPTER TRANSPORT SERVICES',
]

# (model/typecode/icaoaircrafttype tokens, role). First match wins. Only
# ever consulted for an aircraft that already matched FIRE_OPERATORS.
#
# Deliberately manufacturer-name-FREE (just the model token, e.g. '206' not
# 'BELL 206'): the source CSV's `model` field puts the manufacturer in a
# trailing "(Manufacturer)" suffix inconsistently ("206B (Bell)",
# "AS 350 B-2 (Aerospatiale)"), so a phrase requiring word order would miss
# real matches. Matching is done against a punctuation/space-STRIPPED
# haystack (see _norm), so keys here should also be pre-stripped -- write
# them without spaces/hyphens and _classify_role normalizes both sides.
MODEL_ROLE = [
    (('DC10', 'DC987', 'MD87', '747', 'B747'), 'vlat'),
    (('BAE146', 'RJ85', 'AVRO146', 'L188', 'ELECTRA', 'C130', 'HERCULES',
      'EC130', 'P2H', 'P2E', 'P2V'), 'lat'),
    (('AT802', 'AIRTRACTOR', 'AT402', 'AT501'), 'sat'),
    (('S2F', 'S2T', 'TRACKER', 'FIRECAT'), 'sat'),         # CAL FIRE's core airtanker
    (('CL215', 'CL415'), 'scooper'),                        # amphibious scooper
    (('KINGAIR', 'BE200', 'BE20', 'BE300', 'BE9', 'COMMANDER',
      'OV10', 'BRONCO', 'MU2', '690'), 'leadplane_or_airattack'),
    (('S64', 'AIRCRANE', 'CH47', 'CHINOOK', 'S61', '214', '107II',
      'KV107', 'VERTOL234', '234'), 'helitanker_type1'),     # heavy lift
    (('UH1', 'HUEY', '205', '212', '412', 'PUMA', 'SA330', 'CH46'), 'helitanker_type2'),  # medium
    (('206', '407', 'AS350', 'H125', 'ECUREUIL', 'A119', 'KOALA'), 'helitanker_type3'),  # light
]


def _fresh(path, max_age_days):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_days * 86400


def ensure_raw_db(force=False):
    """Download the ~94 MB OpenSky global aircraft metadata CSV if missing
    or stale. Slow and explicit -- never called from the live tracking loop,
    only from build_fleet_index() the first time (or once a month after)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and _fresh(RAW_CSV, RAW_MAX_AGE_DAYS):
        return RAW_CSV
    tmp = RAW_CSV + '.part'
    req = urllib.request.Request(RAW_URL, headers={'User-Agent': 'fire-research'})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, 'wb') as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, RAW_CSV)
    return RAW_CSV


def _norm(s):
    """Upper-case, alphanumeric-only. Model names in the source CSV mix
    formats ("206B (Bell)", "AS 350 B-2 (Aerospatiale)", "S-2F3AT"); stripping
    all punctuation/spaces from both the haystack and the MODEL_ROLE keys
    makes the substring check robust to that without needing word-order-
    sensitive manufacturer-name phrases."""
    return ''.join(ch for ch in (s or '').upper() if ch.isalnum())


def _classify_role(model, typecode, icaoaircrafttype):
    hay = _norm(' '.join(x or '' for x in (model, typecode, icaoaircrafttype)))
    for keys, role in MODEL_ROLE:
        if any(_norm(k) in hay for k in keys):
            return role
    return 'unclassified_fire_operator'


def build_fleet_index(force=False, force_download=False):
    """Filter the ~600k-row global database down to aircraft whose
    operator/owner matches a known wildland-fire operator, and persist that
    small index (typically low thousands of rows, not 94 MB) to disk.
    {icao24: {registration, manufacturer, model, typecode, operator, role}}

    ``force`` re-runs classification against the already-cached raw CSV
    (cheap, seconds) -- use this after editing FIRE_OPERATORS/MODEL_ROLE.
    ``force_download`` additionally re-fetches the ~94 MB raw CSV even if it
    isn't stale yet -- these are kept separate so a classifier tweak doesn't
    silently re-trigger a ~90s download."""
    if not force and _fresh(FLEET_JSON, RAW_MAX_AGE_DAYS):
        with open(FLEET_JSON, encoding='utf-8') as fh:
            return json.load(fh)
    raw = ensure_raw_db(force=force_download)
    fleet = {}
    with open(raw, encoding='utf-8', errors='replace', newline='') as fh:
        for row in csv.DictReader(fh):
            op = (row.get('operator') or '').upper()
            ow = (row.get('owner') or '').upper()
            if not any(k in op or k in ow for k in FIRE_OPERATORS):
                continue
            icao24 = (row.get('icao24') or '').strip().lower()
            if not icao24:
                continue
            fleet[icao24] = {
                'registration': row.get('registration'),
                'manufacturer': row.get('manufacturericao') or row.get('manufacturername'),
                'model': row.get('model'),
                'typecode': row.get('typecode'),
                'operator': row.get('operator') or row.get('owner'),
                'role': _classify_role(row.get('model'), row.get('typecode'), row.get('icaoaircrafttype')),
            }
    os.makedirs(os.path.dirname(FLEET_JSON), exist_ok=True)
    with open(FLEET_JSON, 'w', encoding='utf-8') as fh:
        json.dump(fleet, fh, indent=0)
    return fleet


_FLEET_CACHE = None


def lookup(icao24):
    """{registration, manufacturer, model, typecode, operator, role} for a
    known wildland-fire-operator aircraft, or None. Lazily loads/builds the
    fleet index on first call in this process (fast after that -- an
    in-memory dict, not a per-lookup file/network read). A build/download
    failure propagates as an exception; callers (aircraft_fire_edge) must
    treat that as "registry unavailable, fall back to other identification",
    not "not a firefighting aircraft"."""
    global _FLEET_CACHE
    if _FLEET_CACHE is None:
        _FLEET_CACHE = build_fleet_index()
    return _FLEET_CACHE.get((icao24 or '').strip().lower())


if __name__ == '__main__':
    import sys
    fleet = build_fleet_index(force='--force' in sys.argv, force_download='--redownload' in sys.argv)
    print(f'{len(fleet)} known wildland-fire-operator aircraft cached -> {FLEET_JSON}')
    from collections import Counter
    roles = Counter(v['role'] for v in fleet.values())
    for role, n in roles.most_common():
        print(f'  {role:28s} {n}')
    if len(sys.argv) > 1 and sys.argv[1].startswith('0x'):
        print(lookup(sys.argv[1][2:]))
