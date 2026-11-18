import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from model_assets import inspect_assets, sha256
from pipeline import resumable
from recording_fixture import make_dataset

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'runner/build_maps.py'


class ProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def execute(self, *args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=self.root,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                              capture_output=True, text=True)

    def build(self, *extra, calibration=None):
        return self.execute(
            RUNNER, '--bag', self.root / 'input/bag', '--output-root', self.root / 'out',
            '--calibration', calibration or self.root / 'input/extrinst.txt',
            '--prepare-only', *extra,
        )

    def test_real_mcap_prepare_and_resume_without_models(self):
        make_dataset(self.root / 'input')
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run, = (self.root / 'out').iterdir()
        report = json.loads((run / 'delivery/MAPPING_RESULT.json').read_text())
        self.assertEqual(report['status'], 'prepared_not_mapped')
        self.assertFalse(report['success'])
        self.assertFalse((run / 'native_maps').exists())

        resumed = self.execute(RUNNER, '--resume', run, '--prepare-only')
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        owner = json.loads((run / 'AUTOBUILD_RUN.json').read_text())
        Path(owner['calibration']).write_text('changed')
        rejected = self.execute(RUNNER, '--resume', run, '--prepare-only')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('配置/标定已变', rejected.stdout + rejected.stderr)

    def test_optional_labels_are_frozen_for_the_run(self):
        make_dataset(self.root / 'input')
        labels = self.root / 'labels.yaml'
        labels.write_text('version: 3\nannotations:\n  - id: 1\n    name: cabinet\n')
        result = self.build('--label-yaml', labels)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run, = (self.root / 'out').iterdir()
        owner = json.loads((run / 'AUTOBUILD_RUN.json').read_text())
        frozen = Path(owner['label_yaml'])
        self.assertEqual(frozen.read_bytes(), labels.read_bytes())
        self.assertFalse((run / 'native_maps').exists())
        frozen.write_text('changed')
        result = self.execute(RUNNER, '--resume', run, '--prepare-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('冻结的标签YAML已变', result.stdout + result.stderr)

    def test_wrong_recording_calibration_is_rejected(self):
        make_dataset(self.root / 'input')
        source = self.root / 'input/extrinst.txt'
        source.write_text(source.read_text().replace('base_link', 'other_base'))
        result = self.build()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Odometry child_frame', result.stdout + result.stderr)

    def test_asset_verifier_checks_content_and_reports_stage(self):
        root = self.root / 'models'
        root.mkdir()
        model = root / 'model.pt'
        model.write_text('pinned')
        entry = dict(root='models', path=model.name, stages=['hmsg'],
                     source='fixture', bytes=model.stat().st_size, sha256=sha256(model))
        with patch('model_assets.manifest', return_value=[entry]):
            self.assertTrue(inspect_assets(root, self.root, stages={'hmsg'})[0]['passed'])
            model.write_text('edited')
            check, = inspect_assets(root, self.root, stages={'hmsg'})
            self.assertFalse(check['passed'])
            self.assertEqual(check['stages'], ['hmsg'])
            self.assertIn('SHA-256', check['error'])
            model.unlink()
            self.assertIn('missing', inspect_assets(root, self.root, stages={'hmsg'})[0]['error'])

    def test_resume_requires_artifacts_and_signature(self):
        signature = 'input-and-runtime-fingerprint'
        artifacts = {'cloud': {'sha256': 'original'}}
        previous = dict(exit_code=0, signature=signature, artifacts=artifacts)
        self.assertTrue(resumable(previous, signature, artifacts))
        self.assertFalse(resumable(previous, 'changed-input', artifacts))
        self.assertFalse(resumable(previous, signature, {'cloud': {'sha256': 'changed'}}))
