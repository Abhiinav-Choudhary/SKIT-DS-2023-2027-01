const mongoose = require('mongoose');

const measurementsSchema = new mongoose.Schema(
  {
    systolicBP: { type: Number, min: 50, max: 300 },
    diastolicBP: { type: Number, min: 30, max: 200 },
    heartRateBpm: { type: Number, min: 20, max: 300 },
    temperatureC: { type: Number, min: 25, max: 45 },
    oxygenSaturationPercent: { type: Number, min: 0, max: 100 },
    weightKg: { type: Number, min: 0.1, max: 700 },
    heightCm: { type: Number, min: 30, max: 275 },
  },
  { _id: false }
);

measurementsSchema.pre('validate', function () {
  if (
    this.systolicBP != null &&
    this.diastolicBP != null &&
    this.systolicBP <= this.diastolicBP
  ) {
    this.invalidate('systolicBP', 'Systolic blood pressure must exceed diastolic blood pressure');
  }
});

const healthRecordSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    recordedAt: {
      type: Date,
      required: true,
      default: Date.now,
    },
    measurements: {
      type: measurementsSchema,
      required: true,
      validate: {
        validator(value) {
          return value && Object.values(value.toObject()).some((item) => item != null);
        },
        message: 'At least one health measurement is required',
      },
    },
    notes: {
      type: String,
      trim: true,
      maxlength: [500, 'Notes cannot exceed 500 characters'],
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

healthRecordSchema.index({ user: 1, recordedAt: -1 });

module.exports = mongoose.model('HealthRecord', healthRecordSchema);