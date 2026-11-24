import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectionCsv, nextRevision } from '../storage.js';
import { filterDetections, computeSeverity } from '../vision.js';

const inspection = {
  id: 'I-1',
  name: 'lane, "north".jpg',
  width: 1000,
  height: 500,
  createdAt: '2026-01-02T00:00:00Z',
  model: 'YOLOv8s RoadWatch',
  threshold: 0.35,
  candidates: [
    {
      id: 'C-1',
      label: 'Pothole',
      box: [10, 20, 210, 170],
      score: 0.874,
      areaPct: 6.0,
      severity: 'High',
      status: 'confirmed',
    },
  ],
};

test('exports candidate details, area percentage, severity, and review status', () => {
  const [header, row, end] = inspectionCsv(inspection).split('\r\n');
  assert.match(header, /"label","left","top","right","bottom","score","area_pct","severity","review_status"$/);
  assert.equal(
    row,
    '"I-1","lane, ""north"".jpg","1000","500","2026-01-02T00:00:00Z","YOLOv8s RoadWatch","0.35","1","C-1","Pothole","10","20","210","170","0.8740","6.00","High","confirmed"'
  );
  assert.equal(end, '');
});

test('exports empty inspection with blank detection fields', () => {
  const output = inspectionCsv({ ...inspection, candidates: [] });
  assert.equal(output.trimEnd().split('\r\n').length, 2);
  assert.match(output, /"0","","","","","","","","","",""\r\n$/);
});

test('guards spreadsheet formulas in filenames', () => {
  for (const filename of ['=1+1', '+99', '-99', '@cmd', '  =1', '\ttext', '\n=2']) {
    const output = inspectionCsv({ ...inspection, name: filename });
    assert.ok(output.includes('"\'' + filename + '"'));
  }
});

test('increments revisions and rejects stale updates', () => {
  assert.equal(nextRevision(undefined, {}), 1);
  assert.equal(nextRevision({ revision: 1 }, { revision: 1 }), 2);
  assert.throws(() => nextRevision({ revision: 2 }, { revision: 1 }), { name: 'InspectionConflictError' });
});

test('filters detections by threshold and NMS IoU', () => {
  const raw = [
    { box: [0, 0, 100, 100], score: 0.9, classId: 0, label: 'Pothole' },
    { box: [5, 5, 100, 100], score: 0.8, classId: 0, label: 'Pothole' },
    { box: [200, 200, 300, 300], score: 0.3, classId: 0, label: 'Pothole' },
  ];
  const kept = filterDetections(raw, 0.35, 0.45);
  assert.equal(kept.length, 1);
  assert.equal(kept[0].score, 0.9);
});

test('computes severity grades from box area and confidence', () => {
  const high = computeSeverity([0, 0, 300, 200], 1000, 1000, 0.85);
  const low = computeSeverity([0, 0, 40, 40], 1000, 1000, 0.45);
  assert.equal(high.level, 'high');
  assert.equal(low.level, 'low');
});
