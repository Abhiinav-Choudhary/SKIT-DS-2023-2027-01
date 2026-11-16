const HealthRecord = require('../models/HealthRecord');

const WRITABLE_FIELDS = ['recordedAt', 'measurements', 'notes'];

const pickWritableFields = (data) => {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(data).filter(([field]) => WRITABLE_FIELDS.includes(field))
  );
};

/**
 * Create a new health record for the authenticated user
 */
const createHealthRecord = async (userId, data) => {
  const fields = pickWritableFields(data);

  if (
    !fields.measurements ||
    typeof fields.measurements !== 'object' ||
    Array.isArray(fields.measurements)
  ) {
    const error = new Error('Measurements object is required');
    error.statusCode = 400;
    throw error;
  }

  const hasMeasurement = Object.values(fields.measurements).some((val) => val != null);
  if (!hasMeasurement) {
    const error = new Error('At least one health measurement is required');
    error.statusCode = 400;
    throw error;
  }

  if (fields.recordedAt !== undefined) {
    const d = new Date(fields.recordedAt);
    if (isNaN(d.getTime())) {
      const error = new Error('Invalid recordedAt date format');
      error.statusCode = 400;
      throw error;
    }
  }

  const { systolicBP, diastolicBP } = fields.measurements;
  if (
    (systolicBP != null && isNaN(Number(systolicBP))) ||
    (diastolicBP != null && isNaN(Number(diastolicBP)))
  ) {
    const error = new Error('Blood pressure values must be valid numbers');
    error.statusCode = 400;
    throw error;
  }
  if (systolicBP != null && diastolicBP != null && Number(systolicBP) <= Number(diastolicBP)) {
    const error = new Error('Systolic blood pressure must exceed diastolic blood pressure');
    error.statusCode = 400;
    throw error;
  }

  return HealthRecord.create({
    ...fields,
    user: userId,
  });
};

/**
 * List health records for authenticated user with pagination and optional date filters
 * Supports both listHealthRecords(userId, query) and listHealthRecords(userId, page, limit)
 */
const listHealthRecords = async (userId, pageOrQuery = {}, maybeLimit) => {
  let rawPage;
  let rawLimit;
  let startDate;
  let endDate;

  if (maybeLimit !== undefined) {
    rawPage = pageOrQuery;
    rawLimit = maybeLimit;
  } else if (typeof pageOrQuery === 'object' && pageOrQuery !== null) {
    rawPage = pageOrQuery.page;
    rawLimit = pageOrQuery.limit;
    startDate = pageOrQuery.startDate;
    endDate = pageOrQuery.endDate;
  } else {
    rawPage = pageOrQuery;
    rawLimit = undefined;
  }

  const parsedPage =
    rawPage !== undefined && rawPage !== null && rawPage !== ''
      ? Number(rawPage)
      : 1;

  const parsedLimit =
    rawLimit !== undefined && rawLimit !== null && rawLimit !== ''
      ? Number(rawLimit)
      : 20;

  if (
    !Number.isInteger(parsedPage) ||
    parsedPage < 1 ||
    !Number.isInteger(parsedLimit) ||
    parsedLimit < 1 ||
    parsedLimit > 100
  ) {
    const error = new Error('Page must be positive and limit must be between 1 and 100');
    error.statusCode = 400;
    throw error;
  }

  const filter = { user: userId };

  if (startDate || endDate) {
    let start;
    let end;
    filter.recordedAt = {};

    if (startDate) {
      start = new Date(startDate);
      if (isNaN(start.getTime())) {
        const error = new Error('Invalid startDate format');
        error.statusCode = 400;
        throw error;
      }
      filter.recordedAt.$gte = start;
    }
    if (endDate) {
      end = new Date(endDate);
      if (isNaN(end.getTime())) {
        const error = new Error('Invalid endDate format');
        error.statusCode = 400;
        throw error;
      }
      filter.recordedAt.$lte = end;
    }
    if (start && end && start > end) {
      const error = new Error('startDate cannot be after endDate');
      error.statusCode = 400;
      throw error;
    }
  }

  const records = await HealthRecord.find(filter)
    .sort({ recordedAt: -1 })
    .skip((parsedPage - 1) * parsedLimit)
    .limit(parsedLimit);

  let total = records.length;
  if (typeof HealthRecord.countDocuments === 'function') {
    try {
      total = await HealthRecord.countDocuments(filter);
    } catch {
      total = records.length;
    }
  }

  return {
    records,
    page: parsedPage,
    limit: parsedLimit,
    pagination: {
      total,
      page: parsedPage,
      limit: parsedLimit,
      totalPages: Math.ceil(total / parsedLimit) || 1,
    },
  };
};

/**
 * Retrieve single health record by ID (scoped to user)
 */
const getHealthRecord = async (userId, recordId) => {
  const record = await HealthRecord.findOne({ _id: recordId, user: userId });
  if (!record) {
    const error = new Error('Health record not found');
    error.statusCode = 404;
    throw error;
  }
  return record;
};

/**
 * Update an existing health record (scoped to user)
 */
const updateHealthRecord = async (userId, recordId, data) => {
  const updates = pickWritableFields(data);
  if (Object.keys(updates).length === 0) {
    const error = new Error('Provide at least one updatable health record field');
    error.statusCode = 400;
    throw error;
  }

  if (updates.recordedAt !== undefined) {
    const d = new Date(updates.recordedAt);
    if (isNaN(d.getTime())) {
      const error = new Error('Invalid recordedAt date format');
      error.statusCode = 400;
      throw error;
    }
  }

  if (updates.measurements !== undefined) {
    if (
      !updates.measurements ||
      typeof updates.measurements !== 'object' ||
      Array.isArray(updates.measurements)
    ) {
      const error = new Error('Measurements must be an object');
      error.statusCode = 400;
      throw error;
    }

    const existing = await HealthRecord.findOne({ _id: recordId, user: userId });
    if (!existing) {
      const error = new Error('Health record not found');
      error.statusCode = 404;
      throw error;
    }

    const currentMeasurements = existing.measurements?.toObject
      ? existing.measurements.toObject()
      : (existing.measurements || {});
    updates.measurements = { ...currentMeasurements, ...updates.measurements };

    const { systolicBP, diastolicBP } = updates.measurements;
    if (systolicBP != null && diastolicBP != null && Number(systolicBP) <= Number(diastolicBP)) {
      const error = new Error('Systolic blood pressure must exceed diastolic blood pressure');
      error.statusCode = 400;
      throw error;
    }
  }

  const record = await HealthRecord.findOneAndUpdate(
    { _id: recordId, user: userId },
    updates,
    { new: true, runValidators: true }
  );

  if (!record) {
    const error = new Error('Health record not found');
    error.statusCode = 404;
    throw error;
  }

  return record;
};

/**
 * Delete health record (scoped to user)
 */
const deleteHealthRecord = async (userId, recordId) => {
  const record = await HealthRecord.findOneAndDelete({ _id: recordId, user: userId });
  if (!record) {
    const error = new Error('Health record not found');
    error.statusCode = 404;
    throw error;
  }
  return record;
};

/**
 * Generate health vitals summary and analytics for authenticated user
 */
const getHealthSummary = async (userId) => {
  const query = HealthRecord.find({ user: userId }).sort({ recordedAt: -1 });
  let records;
  if (typeof query.then === 'function') {
    records = await query;
  } else if (typeof query.limit === 'function') {
    records = await query.limit(1000);
  } else {
    records = [];
  }

  if (!records || !Array.isArray(records) || records.length === 0) {
    return {
      totalRecords: 0,
      latestRecord: null,
      averages: null,
    };
  }

  const latestRecord = records[0];

  // Calculate averages across available records
  let sumSys = 0, countSys = 0;
  let sumDia = 0, countDia = 0;
  let sumHr = 0, countHr = 0;
  let sumGlucose = 0, countGlucose = 0;
  let sumSpo2 = 0, countSpo2 = 0;
  let sumTemp = 0, countTemp = 0;
  let sumWeight = 0, countWeight = 0;

  for (const rec of records) {
    const rawM = rec.measurements;
    const m = rawM && typeof rawM.toObject === 'function' ? rawM.toObject() : (rawM || {});
    if (m.systolicBP != null) { sumSys += Number(m.systolicBP); countSys++; }
    if (m.diastolicBP != null) { sumDia += Number(m.diastolicBP); countDia++; }
    if (m.heartRateBpm != null) { sumHr += Number(m.heartRateBpm); countHr++; }
    if (m.bloodGlucoseMgDl != null) { sumGlucose += Number(m.bloodGlucoseMgDl); countGlucose++; }
    if (m.oxygenSaturationPercent != null) { sumSpo2 += Number(m.oxygenSaturationPercent); countSpo2++; }
    if (m.temperatureC != null) { sumTemp += Number(m.temperatureC); countTemp++; }
    if (m.weightKg != null) { sumWeight += Number(m.weightKg); countWeight++; }
  }

  return {
    totalRecords: records.length,
    latestRecord,
    averages: {
      systolicBP: countSys > 0 ? Math.round(sumSys / countSys) : null,
      diastolicBP: countDia > 0 ? Math.round(sumDia / countDia) : null,
      heartRateBpm: countHr > 0 ? Math.round(sumHr / countHr) : null,
      bloodGlucoseMgDl: countGlucose > 0 ? Math.round(sumGlucose / countGlucose) : null,
      oxygenSaturationPercent: countSpo2 > 0 ? Math.round(sumSpo2 / countSpo2) : null,
      temperatureC: countTemp > 0 ? Math.round((sumTemp / countTemp) * 10) / 10 : null,
      weightKg: countWeight > 0 ? Math.round((sumWeight / countWeight) * 10) / 10 : null,
    },
  };
};

module.exports = {
  createHealthRecord,
  listHealthRecords,
  getHealthRecord,
  updateHealthRecord,
  deleteHealthRecord,
  getHealthSummary,
};
