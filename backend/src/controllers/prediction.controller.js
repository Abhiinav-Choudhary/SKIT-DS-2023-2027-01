const predictionService = require('../services/prediction.service');

const predictDiabetes = async (req, res, next) => {
  try {
    const prediction = await predictionService.predictDiabetes(req.user._id, req.body);
    res.status(201).json({
      success: true,
      message: 'Diabetes risk prediction completed successfully',
      data: { prediction },
    });
  } catch (error) {
    next(error);
  }
};

const predictHeart = async (req, res, next) => {
  try {
    const prediction = await predictionService.predictHeartDisease(req.user._id, req.body);
    res.status(201).json({
      success: true,
      message: 'Heart disease risk prediction completed successfully',
      data: { prediction },
    });
  } catch (error) {
    next(error);
  }
};

const list = async (req, res, next) => {
  try {
    const result = await predictionService.listPredictions(req.user._id, req.query);
    res.status(200).json({
      success: true,
      message: 'Predictions retrieved successfully',
      data: result,
    });
  } catch (error) {
    next(error);
  }
};

const getById = async (req, res, next) => {
  try {
    const prediction = await predictionService.getPredictionById(req.user._id, req.params.id);
    res.status(200).json({
      success: true,
      message: 'Prediction retrieved successfully',
      data: { prediction },
    });
  } catch (error) {
    next(error);
  }
};

const remove = async (req, res, next) => {
  try {
    const prediction = await predictionService.deletePrediction(req.user._id, req.params.id);
    res.status(200).json({
      success: true,
      message: 'Prediction deleted successfully',
      data: { id: prediction._id || prediction.id },
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  predictDiabetes,
  predictHeart,
  list,
  getById,
  remove,
};

