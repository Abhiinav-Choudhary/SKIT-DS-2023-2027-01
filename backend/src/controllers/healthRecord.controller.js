const healthRecordService = require('../services/healthRecord.service');

const create = async (req, res, next) => {
  try {
    const record = await healthRecordService.createHealthRecord(req.user._id, req.body);
    res.status(201).json({
      success: true,
      message: 'Health record created successfully',
      data: { record },
    });
  } catch (error) {
    next(error);
  }
};

const list = async (req, res, next) => {
  try {
    const result = await healthRecordService.listHealthRecords(
      req.user._id,
      req.query.page,
      req.query.limit
    );
    res.status(200).json({
      success: true,
      message: 'Health records retrieved successfully',
      data: result,
    });
  } catch (error) {
    next(error);
  }
};

const getById = async (req, res, next) => {
  try {
    const record = await healthRecordService.getHealthRecord(req.user._id, req.params.id);
    res.status(200).json({
      success: true,
      message: 'Health record retrieved successfully',
      data: { record },
    });
  } catch (error) {
    next(error);
  }
};

const update = async (req, res, next) => {
  try {
    const record = await healthRecordService.updateHealthRecord(
      req.user._id,
      req.params.id,
      req.body
    );
    res.status(200).json({
      success: true,
      message: 'Health record updated successfully',
      data: { record },
    });
  } catch (error) {
    next(error);
  }
};

const remove = async (req, res, next) => {
  try {
    const record = await healthRecordService.deleteHealthRecord(req.user._id, req.params.id);
    res.status(200).json({
      success: true,
      message: 'Health record deleted successfully',
      data: { id: record._id },
    });
  } catch (error) {
    next(error);
  }
};

module.exports = { create, list, getById, update, remove };