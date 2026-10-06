"""Correct the figure title after completion without changing frozen study code."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_facility_subtype as frozen


def main():
    source = ROOT / 'reports/facility-subtype-study-comparison.json'
    output = source.with_suffix('.png')
    sidecar = source.with_suffix('.plot.json')
    archive = ROOT / 'runs/facility-subtype-original-plot-title'
    frozen.require(not archive.exists(), 'Preserve the original title evidence')
    aggregate = json.loads(source.read_text(encoding='utf-8'))
    previous = json.loads(sidecar.read_text(encoding='utf-8'))
    rows = frozen.measurements(aggregate)
    frozen.require(rows == previous['experiments']
        and previous['plot_sha256'] == frozen.sha(output)
        and previous['source_aggregate_sha256'] == frozen.sha(source),
        'Original figure and actual measurements must agree')
    archive.mkdir()
    for path in (output, sidecar):
        (archive / path.name).write_bytes(path.read_bytes())
    corrected = archive / 'corrected.png'
    title = 'Spalling-negative subtype sampling on source validation'
    frozen.render_plot(rows, corrected, title=title)
    note = {'reason': 'This study changes sampling frequency; the original title incorrectly said pixel comparison',
        'performed_after_completed_training': True, 'new_training_epochs': 0,
        'original_plot_sha256': previous['plot_sha256'],
        'original_plot_record_sha256': frozen.sha(archive / sidecar.name),
        'correction_script_sha256': frozen.sha(Path(__file__)),
        'frozen_plot_script_sha256': frozen.sha(Path(frozen.__file__)),
        'original_and_corrected_measurements_exactly_equal': True, 'corrected_title': title}
    corrected.replace(output)
    previous.update(plot_sha256=frozen.sha(output), title_correction=note)
    sidecar.write_bytes((json.dumps(previous, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    print({'status':'title_corrected','measured_values_changed':False,'new_epochs':0})


if __name__ == '__main__': main()
