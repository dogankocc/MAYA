#pragma once

#include <string>

namespace llm::server {

// ========== MSSQL Veritabani Semasi ==========
// Bu dosya ModelManager tarafindan kullanilir.
// Ayrica disariya export edilir ve "schema.sql" olarak kaydedilebilir.

constexpr const char* kModelSchemaSql = R"SQL(
-- ============================================
-- MAYA LLM Model Yonetim Sistemi - MSSQL Semasi
-- ============================================

-- Eger tablolar varsa once sil (development icin)
-- NOT: Production'da bunu kullanmayin!
/*
DROP TABLE IF EXISTS TrainingRuns;
DROP TABLE IF EXISTS Checkpoints;
DROP TABLE IF EXISTS Models;
*/

-- ============================================
-- 1. Models Tablosu
-- Temel model bilgilerini tutar
-- ============================================
CREATE TABLE Models (
    Id NVARCHAR(32) PRIMARY KEY,                    -- 16 karakterli hex ID
    Name NVARCHAR(255) NOT NULL,                     -- Model adi (orn: "maya-code-v1")
    Version NVARCHAR(64) NOT NULL DEFAULT 'latest',  -- Versiyon (orn: "2026-09-18", "v1.0")
    Path NVARCHAR(500) NOT NULL,                     -- Local dosya yolu
    FileSizeBytes BIGINT NOT NULL DEFAULT 0,         -- Dosya boyutu (byte)
    IsQuantized BIT NOT NULL DEFAULT 0,              -- INT8 quantize edilmis mi?

    -- Mimari Parametreler
    NumLayers INT NOT NULL DEFAULT 0,
    HiddenDim INT NOT NULL DEFAULT 0,
    NumHeads INT NOT NULL DEFAULT 0,
    NumKvHeads INT NOT NULL DEFAULT 0,
    IntermediateDim INT NOT NULL DEFAULT 0,
    MaxSeqLen INT NOT NULL DEFAULT 0,
    VocabSize INT NOT NULL DEFAULT 8000,

    -- Preset Bilgisi
    Preset NVARCHAR(32) NOT NULL DEFAULT 'Custom',   -- 'Tiny', 'Small', 'Medium', 'Large', 'XLarge', 'Custom'

    -- Kapasite Bilgisi
    TotalParameters BIGINT NOT NULL DEFAULT 0,
    ActiveParameters BIGINT NOT NULL DEFAULT 0,
    ApproxGbFp32 FLOAT NOT NULL DEFAULT 0.0,
    ApproxGbInt8 FLOAT NOT NULL DEFAULT 0.0,
    ComplexityLevel NVARCHAR(64) NOT NULL DEFAULT 'Unknown',

    -- Egitim Bilgisi
    TrainedSteps BIGINT NOT NULL DEFAULT 0,
    FinalLoss FLOAT NOT NULL DEFAULT 0.0,
    TrainedCorpus NVARCHAR(500),

    -- Zaman Damgalari
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    -- Indexler
    INDEX IX_Models_Name (Name),
    INDEX IX_Models_CreatedAt (CreatedAt DESC),
    INDEX IX_Models_Preset (Preset)
);

-- ============================================
-- 2. Checkpoints Tablosu
-- Her modelin kayit noktalarini tutar
-- ============================================
CREATE TABLE Checkpoints (
    Id NVARCHAR(32) PRIMARY KEY,
    ModelId NVARCHAR(32) NOT NULL FOREIGN KEY REFERENCES Models(Id) ON DELETE CASCADE,

    Path NVARCHAR(500) NOT NULL,
    FileSizeBytes BIGINT NOT NULL DEFAULT 0,
    IsQuantized BIT NOT NULL DEFAULT 0,

    Step BIGINT NOT NULL DEFAULT 0,
    Loss FLOAT NOT NULL DEFAULT 0.0,
    LearningRate FLOAT NOT NULL DEFAULT 0.0,

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    INDEX IX_Checkpoints_ModelId (ModelId),
    INDEX IX_Checkpoints_Step (ModelId, Step DESC)
);

-- ============================================
-- 3. TrainingRuns Tablosu
-- Egitim calismalarini takip eder
-- ============================================
CREATE TABLE TrainingRuns (
    Id NVARCHAR(32) PRIMARY KEY,
    ModelId NVARCHAR(32) NOT NULL FOREIGN KEY REFERENCES Models(Id) ON DELETE CASCADE,

    Status NVARCHAR(32) NOT NULL DEFAULT 'Pending',  -- 'Pending', 'Running', 'Completed', 'Failed', 'Paused'
    CorpusPath NVARCHAR(500),
    CorpusSamples BIGINT NOT NULL DEFAULT 0,

    -- Egitim Parametreleri
    MaxSteps BIGINT NOT NULL DEFAULT 10000,
    CurrentStep BIGINT NOT NULL DEFAULT 0,
    LearningRate FLOAT NOT NULL DEFAULT 3e-4,
    BatchSize INT NOT NULL DEFAULT 8,
    WarmupSteps INT NOT NULL DEFAULT 0,
    WeightDecay FLOAT NOT NULL DEFAULT 0.01,

    -- Quantization
    QuantizeAfterTraining BIT NOT NULL DEFAULT 1,

    -- Sonuclar
    FinalLoss FLOAT,
    BestLoss FLOAT,
    BestStep BIGINT,

    -- Upload
    UploadToGoogleDrive BIT NOT NULL DEFAULT 0,
    GoogleDriveFolderId NVARCHAR(128),
    UploadStatus NVARCHAR(32),  -- 'Pending', 'Completed', 'Failed'

    -- Hata
    ErrorMessage NVARCHAR(MAX),

    -- Zaman Damgalari
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    StartedAt DATETIME2,
    CompletedAt DATETIME2,
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    INDEX IX_TrainingRuns_ModelId (ModelId),
    INDEX IX_TrainingRuns_Status (Status),
    INDEX IX_TrainingRuns_CreatedAt (CreatedAt DESC)
);

-- ============================================
-- 4. Basit Ornek Veriler (Development)
-- ============================================

-- Medium (onerilen) preset icin ornek model kaydi
/*
INSERT INTO Models (
    Id, Name, Version, Path,
    NumLayers, HiddenDim, NumHeads, NumKvHeads, IntermediateDim, MaxSeqLen, VocabSize,
    Preset,
    TotalParameters, ActiveParameters, ApproxGbFp32, ApproxGbInt8, ComplexityLevel,
    CreatedAt, UpdatedAt
)
VALUES (
    '0123456789abcdef',
    'maya-demo-medium',
    'latest',
    'models/maya-demo-medium.ckpt',
    24, 768, 12, 6, 3072, 512, 8000,
    'Medium',
    350000000, 350000000, 1.3, 0.35, 'Orta (1B)',
    GETUTCDATE(), GETUTCDATE()
);
*/

PRINT 'Schema olusturuldu: Models, Checkpoints, TrainingRuns tablolari';
)SQL";

inline std::string GetDefaultSchemaSql() {
    return std::string(kModelSchemaSql);
}

} // namespace llm::server
