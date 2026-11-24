const databaseName = 'campus-roadwatch-v2';
const storeName = 'inspections';

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    let blocked = false;
    request.onupgradeneeded = () => {
      request.result.createObjectStore(storeName, { keyPath: 'id' });
    };
    request.onerror = () => reject(request.error);
    request.onblocked = () => {
      blocked = true;
      reject(new Error('Close other RoadWatch tabs and try again.'));
    };
    request.onsuccess = () => {
      if (blocked) request.result.close();
      else resolve(request.result);
    };
  });
}

async function runTransaction(mode, operation) {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(storeName, mode);
    const committed = new Promise((resolve, reject) => {
      transaction.oncomplete = resolve;
      transaction.onabort = () => reject(transaction.error || new Error('Storage operation was interrupted.'));
    });
    const [result] = await Promise.all([operation(transaction.objectStore(storeName)), committed]);
    return result;
  } finally {
    database.close();
  }
}

function readRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export function nextRevision(current, incoming) {
  const revision = incoming.revision ?? 0;
  if ((current?.revision ?? 0) !== revision) {
    const error = new Error('This inspection was updated in another tab. Reopen it from History.');
    error.name = 'InspectionConflictError';
    throw error;
  }
  return revision + 1;
}

export async function saveInspection(record) {
  const savedRevision = await runTransaction('readwrite', async store => {
    const current = await readRequest(store.get(record.id));
    const revision = nextRevision(current, record);
    await readRequest(store.put({ ...record, revision }));
    return revision;
  });
  record.revision = savedRevision;
}

export async function deleteInspection(id) {
  await runTransaction('readwrite', store => readRequest(store.delete(id)));
}

function timestamp(record) {
  const value = new Date(record.createdAt).getTime();
  return Number.isFinite(value) ? value : 0;
}

export async function listInspections() {
  const records = await runTransaction('readonly', store => readRequest(store.getAll()));
  return records.sort((left, right) => timestamp(right) - timestamp(left));
}

const columns = [
  'inspection_id', 'filename', 'image_width', 'image_height', 'created_at',
  'model', 'threshold', 'candidate_count', 'candidate_id', 'label',
  'left', 'top', 'right', 'bottom', 'score', 'area_pct', 'severity', 'review_status',
];

function csvCell(value) {
  let text = String(value ?? '');
  if (typeof value === 'string' && (/^[\s\uFEFF]*[=+@\-]/u.test(text) || /^[\t\r\n]/u.test(text))) {
    text = "'" + text;
  }
  return '"' + text.replaceAll('"', '""') + '"';
}

export function inspectionCsv(record) {
  const metadata = [
    record.id, record.name, record.width, record.height, record.createdAt,
    record.model, record.threshold, record.candidates.length,
  ];
  const rows = record.candidates.map(candidate => [
    ...metadata,
    candidate.id,
    candidate.label ?? 'Pothole',
    ...candidate.box.map(v => Math.round(v)),
    Number(candidate.score).toFixed(4),
    candidate.areaPct != null ? Number(candidate.areaPct).toFixed(2) : '',
    candidate.severity ?? '',
    candidate.status,
  ]);
  if (rows.length === 0) {
    rows.push([...metadata, '', '', '', '', '', '', '', '', '', '']);
  }
  return [columns, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
}

export async function exportInspection(record) {
  const current = await runTransaction('readonly', store => readRequest(store.get(record.id)));
  if (!current) throw new Error('Saved inspection not found.');
  nextRevision(current, record);
  const contents = new Blob(['\uFEFF', inspectionCsv(current)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(contents);
  const link = document.createElement('a');
  const safeId = String(record.id).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 100);
  link.href = url;
  link.download = 'roadwatch-' + (safeId || 'report') + '.csv';
  try {
    document.body.append(link);
    link.click();
  } finally {
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
