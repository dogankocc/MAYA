-- ============================================
-- MAYA LLM Model Yonetim Sistemi - MSSQL Semasi
-- ============================================
-- Bu semayi MSSQL'de calistirarak tablolari olusturabilirsiniz:
-- sqlcmd -S localhost\SQLEXPRESS -d MAYA_MODELS -i schema.sql

-- ============================================
-- ONCEKI TABLOLARI SIL (SADECE DEVELOPMENT ICIN!)
-- ============================================
/*
DROP TABLE IF EXISTS TrainingRuns;
DROP TABLE IF EXISTS Checkpoints;
DROP TABLE IF EXISTS Models;
*/

-- ============================================
-- 1. Models Tablosu - Temel model bilgileri
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

    -- Preset Bilgisi: 'Tiny', 'Small', 'Medium', 'Large', 'XLarge', 'Custom'
    Preset NVARCHAR(32) NOT NULL DEFAULT 'Custom',

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
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

-- Indexler
CREATE INDEX IX_Models_Name ON Models(Name);
CREATE INDEX IX_Models_CreatedAt ON Models(CreatedAt DESC);
CREATE INDEX IX_Models_Preset ON Models(Preset);

-- ============================================
-- 2. Checkpoints Tablosu - Kayit noktalari
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

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

CREATE INDEX IX_Checkpoints_ModelId ON Checkpoints(ModelId);
CREATE INDEX IX_Checkpoints_Step ON Checkpoints(ModelId, Step DESC);

-- ============================================
-- 3. TrainingRuns Tablosu - Egitim takibi
-- ============================================
CREATE TABLE TrainingRuns (
    Id NVARCHAR(32) PRIMARY KEY,
    ModelId NVARCHAR(32) NOT NULL FOREIGN KEY REFERENCES Models(Id) ON DELETE CASCADE,

    -- Durum: 'Pending', 'Running', 'Completed', 'Failed', 'Paused'
    Status NVARCHAR(32) NOT NULL DEFAULT 'Pending',
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

    -- Google Drive Upload
    UploadToGoogleDrive BIT NOT NULL DEFAULT 0,
    GoogleDriveFolderId NVARCHAR(128),
    UploadStatus NVARCHAR(32),  -- 'Pending', 'Completed', 'Failed'

    -- Hata Mesaji
    ErrorMessage NVARCHAR(MAX),

    -- Zaman Damgalari
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    StartedAt DATETIME2,
    CompletedAt DATETIME2,
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

CREATE INDEX IX_TrainingRuns_ModelId ON TrainingRuns(ModelId);
CREATE INDEX IX_TrainingRuns_Status ON TrainingRuns(Status);
CREATE INDEX IX_TrainingRuns_CreatedAt ON TrainingRuns(CreatedAt DESC);

-- ============================================
-- ORNEK VERI (Test icin)
-- ============================================

/*
-- Medium (onerilen) preset icin ornek model
INSERT INTO Models (
    Id, Name, Version, Path,
    NumLayers, HiddenDim, NumHeads, NumKvHeads, IntermediateDim, MaxSeqLen, VocabSize,
    Preset,
    TotalParameters, ActiveParameters, ApproxGbFp32, ApproxGbInt8, ComplexityLevel,
    CreatedAt, UpdatedAt
)
VALUES (
    NEWID(),  -- SQL Server kendi uretsin
    'maya-demo-medium',
    'latest',
    'models/maya-demo-medium.ckpt',
    24, 768, 12, 6, 3072, 512, 8000,
    'Medium',
    350000000, 350000000, 1.3, 0.35, 'Orta (1B)',
    GETUTCDATE(), GETUTCDATE()
);
*/

PRINT '============================================================';
PRINT '  MAYA LLM Model Yonetim Sistemi Semasi OLUSTURULDU';
PRINT '============================================================';
PRINT '';
PRINT 'Olusturulan tablolar:';
PRINT '  - Models         : Temel model bilgileri';
PRINT '  - Checkpoints    : Kayit noktalari';
PRINT '  - TrainingRuns   : Egitim calismalari takibi';
PRINT '';
PRINT 'Kullanilan presetler:';
PRINT '  - Tiny    : 4 katman, 128 dim  (~10M parametre)';
PRINT '  - Small   : 12 katman, 384 dim (~100M parametre)';
PRINT '  - Medium  : 24 katman, 768 dim (~350M parametre) - ONERILEN';
PRINT '  - Large   : 32 katman, 1024 dim (~1.2B parametre)';
PRINT '  - XLarge  : 40 katman, 2048 dim (~7B+ parametre)';
PRINT '';
PRINT '============================================================';
