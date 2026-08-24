================================================================================
include/llm/training - EGITIM (Adim 11)
================================================================================

Model egitimi: forward + backward + AdamW.
Namespace: llm::training


================================================================================
DOSYALAR
================================================================================

loss.hpp              CrossEntropyLoss (next-token prediction)
autograd.hpp          MatMul/Add/Mul/Silu/RmsNorm backward
adamw.hpp             AdamW optimizer
parameter.hpp         ParameterList (model agirliklari + grad)
transformer_train.hpp TrainStep (tek adim forward+backward)
trainer.hpp           Trainer (TrainStep + TrainEpoch)


================================================================================
AKIS
================================================================================

  1. Forward: embedding -> transformer blocks -> RMSNorm -> lm_head
  2. Loss: cross-entropy (logits[t] -> token[t+1])
  3. Backward: tum katmanlardan gradyan toplama
  4. AdamW: agirlik guncelleme


================================================================================
ORNEK
================================================================================

  llm::training::TrainerConfig config;
  config.optimizer.learningRate = 0.01f;
  llm::training::Trainer trainer(model, config);

  float loss = 0.0f;
  trainer.TrainStep({3, 4, 5, 6}, loss);

  MAYA train --output model.ckpt --steps 20


================================================================================
IMPLEMENTASYON
================================================================================

  src/training/parameter.cpp
  src/training/loss.cpp
  src/training/autograd.cpp
  src/training/adamw.cpp
  src/training/transformer_train.cpp
  src/training/trainer.cpp

================================================================================
