const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'Prediction must belong to a user'],
      index: true,
    },
    modelType: {
      type: String,
      enum: {
        values: ['diabetes', 'heart'],
        message: '{VALUE} is not a supported prediction model type',
      },
      required: [true, 'Model type is required'],
    },
    inputs: {
      type: mongoose.Schema.Types.Mixed,
      required: [true, 'Inference inputs are required'],
    },
    predictedClass: {
      type: Number,
      required: [true, 'Predicted class is required'],
    },
    riskScore: {
      type: Number,
      required: [true, 'Risk score is required'],
      min: [0, 'Risk score cannot be less than 0'],
      max: [100, 'Risk score cannot exceed 100'],
    },
    riskLevel: {
      type: String,
      enum: {
        values: ['Low', 'Mild', 'Moderate', 'High', 'Critical', 'Elevated'],
        message: '{VALUE} is not a recognized risk level',
      },
      required: [true, 'Risk level is required'],
    },
    probabilities: {
      type: mongoose.Schema.Types.Mixed,
    },
    probability: {
      type: Number,
    },
    recommendation: {
      type: String,
      trim: true,
    },
    modelName: {
      type: String,
      trim: true,
    },
    modelVersion: {
      type: String,
      trim: true,
    },
    recordedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
    toJSON: {
      transform(doc, ret) {
        ret.id = ret._id;
        delete ret._id;
        delete ret.__v;
        return ret;
      },
    },
  }
);

predictionSchema.index({ user: 1, createdAt: -1 });
predictionSchema.index({ user: 1, modelType: 1 });

module.exports = mongoose.model('Prediction', predictionSchema);

