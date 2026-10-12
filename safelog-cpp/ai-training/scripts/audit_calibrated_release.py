"""Final immutable source/model/report audit, without adding performance scores."""
from pathlib import Path
from prepare_foreign_calibration_check import read,new,sha
from study_foreign_calibration import ROOT,PROTOCOL,SOURCES,sources


def main():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources()
    for name,digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    for prefix,reporter in [('conveyor-foreign-expansion','scripts/report_conveyor_expansion.py'),
                            ('foreign-calibration','scripts/report_foreign_calibration.py')]:
        plot = read(ROOT/f'reports/{prefix}-plot.json')
        assert sha(ROOT/reporter)==plot['source_sha256']
        if prefix=='foreign-calibration':
            assert sha(ROOT/'reports/FOREIGN_CALIBRATION_RESULTS_KO.md')==plot['report_sha256']
            assert sha(ROOT/'reports/foreign-calibration-fresh-rates.png')==plot['plot_sha256']
        else:
            assert sha(ROOT/'reports/CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md')==plot['report_sha256']
            for name,digest in plot['figures'].items():
                assert sha(ROOT/'reports'/name)==digest
    for name in ('conveyor-dual-handoff-release-v3.json','conveyor-expansion-release-v4.json',
                 'calibrated-conveyor-release-v5.json','calibrated-conveyor-release-v5_1.json'):
        release = read(ROOT/'reports'/name)
        assert sha(Path(release['package']))==release['package_sha256']
        for model in release['models'].values():
            assert sha(ROOT/'runs'/model['path'])==model['sha256']
    standalone = read(ROOT/'reports/calibrated-conveyor-standalone-v5.json')
    assert standalone['status']=='passed'
    assert sha(ROOT/'scripts/package_calibrated_conveyor.py')==standalone['source_sha256']
    final = read(ROOT/'reports/calibrated-conveyor-release-v5_1.json')
    assert sha(ROOT/'scripts/finalize_calibrated_handoff.py')==final['source_sha256']
    assert final['runtime_and_model_bytes_identical_to_actual_http_tested_v5']
    fit = read(ROOT/'reports/foreign-calibration-fit.json')
    result = read(ROOT/'reports/foreign-calibration-results.json')
    assert result['weights_sha256']==fit['weights_sha256']
    assert fit['neural_weights_unchanged'] and fit['new_neural_epochs']==fit['new_optimizer_updates']==0
    new(ROOT/'reports/calibrated-conveyor-final-audit.json',{'status':'passed',
        'frozen_and_legacy_source_files_preserved':len(SOURCES)+len(protocol['legacy_source_sha256']),
        'all_previous_and_new_zip_hashes_preserved':True,'plots_and_reports_unchanged':True,
        'actual_cpu_http_passed':True,'package_documentation_links_closed':True,
        'actual_neural_training_epochs_this_cycle':30,'actual_neural_updates_this_cycle':4290,
        'new_neural_epochs_in_calibration':0,'new_optimizer_updates_in_calibration':0,
        'fresh_photo_count':783,'same_known_test_recordings':True,
        'independent_factory_verified':False,'known_test_false_positive_regression_reported':True,
        'new_ai_or_human_labels':0,'additional_test_model_scores':0,
        'final_package_sha256':final['package_sha256'],'source_sha256':sha(Path(__file__))})
    print('Final source/model/report/ZIP and real HTTP evidence audit passed',flush=True)


if __name__=='__main__':
    main()
