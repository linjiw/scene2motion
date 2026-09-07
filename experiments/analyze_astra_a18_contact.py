"""Close bounded A18 at the frozen first-failure gate; never reclassify it."""
import argparse
import json
from pathlib import Path
from experiments.astra_a18_g1_contact import CASES, detection_table, validate
from scene2motion.lean_supervisor import digest, save_new


def analyze(out):
    completed, files = [], [out/'identity.json']
    for case, mode in CASES:
        here = out/f'{case}_{mode}'
        if not (here/'launch.json').exists():
            continue
        try:
            validate(out, case, mode)
        except ValueError:
            # A failed gate is a result only when a complete immutable verification exists.
            if not (here/'verification.json').exists():
                raise
        v = json.loads((here/'verification.json').read_text())
        p = json.loads((here/'process.json').read_text())
        if p['status'] != 'complete':
            raise ValueError('incomplete instrumentation launch')
        completed.append(v)
        files.extend(here/n for n in ('launch.json', 'process.json', 'verification.json',
                                     'eval/contacts.json', 'eval/states.npz'))
    if [(v['case'], v['logging']) for v in completed] != [('positive', 'off'), ('positive', 'base')]:
        raise ValueError('expected bounded two-job stop')
    raw = json.loads((out/'positive_base/eval/contacts.json').read_text())
    table = detection_table(raw)
    if table != completed[1]['detections'] or completed[1]['pass']:
        raise ValueError('raw detection table or closure changed')
    result = {'schema_version': 'astra-a18-bounded-closure-v1', 'status': 'closed_failed_force_gate',
        'jobs_launched': 2, 'operational_instrument_exposures': 64, 'jobs_unlaunched': 4,
        'target_placements_with_points': sum(r['target_point_observations'] > 0 for r in table),
        'target_placements_with_nonzero_reported_impulse': sum(r['target_normal_impulse_ns'] > 0 for r in table),
        'observer_states_exact': completed[1]['checks']['observer_states_exact'],
        'force_sensitivity_validated': False, 'specificity_tested': False, 'capacity_doubling_tested': False,
        'whole_body_contact_validated': False, 'detection_table': table,
        'interpretation': 'Points were reported on all32 induced-overlap placements, but reported impulses were zero. Force sensitivity remains unresolved; this does not distinguish stimulus insufficiency from reader behavior.',
        'source_hashes': {str(p): digest(p) for p in files}}
    path = out/'summary.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('closure changed')
    else:
        save_new(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('detection_table', 'source_hashes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    analyze(parser.parse_args().out.resolve())
