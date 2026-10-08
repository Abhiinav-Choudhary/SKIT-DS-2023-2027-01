const HealthRecord = require('../models/HealthRecord');

const writableFields = ['recordedAt', 'measurements', 'notes'];

const pickWritableFields = (data) => {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(data).filter(([field]) => writableFields.includes(field))
  );
};

const createHealthRecord = async (userId, data) => {
  return HealthRecord.create({
    ...pickWritableFields(data),
    user: userId,
  });
};

const listHealthRecords = async (userId, page, limit) => {
  const parsedPage = Number(page || 1);
  const parsedLimit = Number(limit || 20);

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

  const records = await HealthRecord.find({ user: userId })
    .sort({ recordedAt: -1 })
    .skip((parsedPage - 1) * parsedLimit)
    .limit(parsedLimit);

  return { records, page: parsedPage, limit: parsedLimit };
};

const getHealthRecord = async (userId, recordId) => {
  const record = await HealthRecord.findOne({ _id: recordId, user: userId });
  if (!record) {
    const error = new Error('Health record not found');
    error.statusCode = 404;
    throw error;
  }
  return record;
};

const updateHealthRecord = async (userId, recordId, data) => {
  const updates = pickWritableFields(data);
  if (Object.keys(updates).length === 0) {
    const error = new Error('Provide at least one updatable health record field');
    error.statusCode = 400;
    throw error;
  }

  if (updates.measurements) {
    const existing = await HealthRecord.findOne({ _id: recordId, user: userId });
    if (!existing) {
      const error = new Error('Health record not found');
      error.statusCode = 404;
      throw error;
    }

    const currentMeasurements = existing.measurements?.toObject
      ? existing.measurements.toObject()
      : existing.measurements;
    updates.measurements = { ...currentMeasurements, ...updates.measurements };

    if (
      updates.measurements.systolicBP != null &&
      updates.measurements.diastolicBP != null &&
      updates.measurements.systolicBP <= updates.measurements.diastolicBP
    ) {
      const error = new Error(
        'Systolic blood pressure must exceed diastolic blood pressure'
      );
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

const deleteHealthRecord = async (userId, recordId) => {
  const record = await HealthRecord.findOneAndDelete({ _id: recordId, user: userId });
  if (!record) {
    const error = new Error('Health record not found');
    error.statusCode = 404;
    throw error;
  }
  return record;
};

module.exports = {
  createHealthRecord,
  listHealthRecords,
  getHealthRecord,
  updateHealthRecord,
  deleteHealthRecord,
};