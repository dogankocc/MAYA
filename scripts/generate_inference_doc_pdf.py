#!/usr/bin/env python3
"""Generate inference pipeline documentation PDF in project documents/."""

import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "documents"
OUTPUT = OUTPUT_DIR / "CMakeProject2_Inference_Akisi.pdf"
FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")


class DocPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(18, 18, 18)

    def usable_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Sayfa {self.page_no()}", align="C")


def section_title(pdf: DocPDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font("ArialB", "", 13)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(pdf.usable_width(), 8, text)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def body(pdf: DocPDF, text: str) -> None:
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(pdf.usable_width(), 5.5, text)
    pdf.ln(1)


def code_block(pdf: DocPDF, text: str, size: float = 7.0) -> None:
    pdf.set_font("Mono", "", size)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(pdf.usable_width(), 3.8, text, fill=True)
    pdf.set_font("Arial", "", 10)
    pdf.ln(2)


def table_row(pdf: DocPDF, col1: str, col2: str) -> None:
    w = pdf.usable_width()
    c1 = w * 0.38
    c2 = w * 0.62
    y0 = pdf.get_y()
    pdf.set_font("Arial", "", 9)
    pdf.set_xy(pdf.l_margin, y0)
    pdf.multi_cell(c1, 5, col1, border=1)
    y1 = pdf.get_y()
    pdf.set_xy(pdf.l_margin + c1, y0)
    pdf.multi_cell(c2, 5, col2, border=1)
    y2 = pdf.get_y()
    pdf.set_y(max(y1, y2))


def bullet(pdf: DocPDF, text: str) -> None:
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(pdf.usable_width(), 5.5, f"  - {text}")


def build_pdf() -> Path:
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    if not FONT_REGULAR.exists():
        raise FileNotFoundError(f"Font bulunamadi: {FONT_REGULAR}")

    pdf.add_font("Arial", "", str(FONT_REGULAR))
    pdf.add_font("ArialB", "", str(FONT_BOLD if FONT_BOLD.exists() else FONT_REGULAR))
    pdf.add_font("Mono", "", str(FONT_MONO if FONT_MONO.exists() else FONT_REGULAR))

    pdf.set_font("ArialB", "", 18)
    pdf.multi_cell(pdf.usable_width(), 10, "CMakeProject2 - Model Dosyasindan Cevap Uretimi")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(pdf.usable_width(), 6, "Egitim sonrasi inference akisi - adim adim aciklama")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    body(
        pdf,
        "Bu dokuman, egitimle olusan model.ckptq dosyasinin sunucu tarafinda nasil "
        "yuklendigini ve kullanici mesajindan Turkce cevaba nasil donusturuldugunu aciklar.",
    )

    section_title(pdf, "Ozet: Egitim vs Kullanim")
    bullet(pdf, "Egitim: veri okunur, loss hesaplanir, agirliklar guncellenir, model.ckpt yazilir.")
    bullet(pdf, "Kullanim: agirliklar dosyadan yuklenir, sabit tutulur, her mesajda token token tahmin yapilir.")
    bullet(pdf, "Iki dosya gerekir: model.ckptq (agirliklar) + tokenizer_data/ (metin <-> sayi sozlugu).")

    section_title(pdf, "Ornek Istek")
    body(pdf, "Flutter uygulamasi POST /api/v1/chat ile JSON gonderir:")
    code_block(pdf, '{"prompt": "Merhaba", "max_tokens": 64, "temperature": 0.7}')

    section_title(pdf, "Adim 0 - Sunucu acilinca model yuklenir")
    body(
        pdf,
        "serve komutu calisinca HybridChatBackend olusur. model.ckptq ve tokenizer_data "
        "RAM'e alinir. Checkpoint header ve tensor agirliklari okunur; TransformerModel insa edilir.",
    )

    section_title(pdf, "Adim 1 - HTTP istegi")
    body(pdf, "ChatHttpServer POST /api/v1/chat JSON govdesini parse eder.")

    section_title(pdf, "Adim 2 - Backend secimi")
    body(pdf, "HybridChatBackend local veya openai backend'e yonlendirir.")

    section_title(pdf, "Adim 3 - Intent + prompt sablonu")
    body(
        pdf,
        "Ornek: Niyet: sohbet. Kullanici: Merhaba Asistan:\n"
        "Model egitimde 'Asistan:' sonrasini ogrenmisti.",
    )

    section_title(pdf, "Adim 4 - Metin -> token ID")
    body(pdf, "BpeTokenizer metni sayilara cevirir.")

    section_title(pdf, "Adim 5 - Prefill")
    body(pdf, "Prompt 124 katmandan gecer; son pozisyondan logits (her token icin skor) uretilir.")

    section_title(pdf, "Adim 6 - Sampling")
    body(pdf, "Temperature, top-k, top-p ile bir sonraki token secilir.")

    section_title(pdf, "Adim 7 - Decode dongusu")
    body(pdf, "Her yeni token icin forward + yeni logits; maxTokens veya EOS'a kadar.")

    section_title(pdf, "Adim 8 - Token -> metin")
    body(pdf, "Yeni tokenlar tokenizer ile Turkce metne cevrilir.")

    section_title(pdf, "Adim 9 - Temizle + JSON")
    body(pdf, 'Ornek cevap: {"text":"Selam!","intent":"sohbet","error":null}')

    section_title(pdf, "Gorsel Akis")
    code_block(
        pdf,
        "JSON -> ParseChatRequest -> Intent + prompt\n"
        "-> Encode -> Prefill (124 katman) -> logits\n"
        "-> Sample -> Decode dongusu -> Decode metin\n"
        "-> Sanitize -> JSON -> Flutter",
    )

    pdf.add_page()
    section_title(pdf, "Tam Kod (Inference Pipeline)")
    body(
        pdf,
        "Asagida uctan uca zincirin tam fonksiyonlari var. Transformer katman detayi "
        "(transformer.cpp, attention.cpp) ayri - onlar dosyadaki agirliklarla matris carpimi yapar.",
    )

    section_title(pdf, "1. Model yukleme - chat_session.cpp")
    code_block(
        pdf,
        "Status ChatSession::Load(const std::string& modelPath,\n"
        "                        const std::string& tokenizerPath) {\n"
        "  auto model = LoadModel(modelPath);  // .ckptq -> QuantCheckpoint::LoadRuntime\n"
        "  if (!model.IsOk()) {\n"
        "    return Status::Fail(model.GetError().code, model.GetError().message);\n"
        "  }\n"
        "  auto tokenizer = BpeTokenizer::Load(tokenizerPath);\n"
        "  if (!tokenizer.IsOk()) {\n"
        "    return Status::Fail(tokenizer.GetError().code, tokenizer.GetError().message);\n"
        "  }\n"
        "  model_ = std::move(model.Value());\n"
        "  tokenizer_ = std::move(tokenizer.Value());\n"
        "  engine_.emplace(*model_);\n"
        "  generator_.emplace(*engine_, sampling_);\n"
        "  loaded_ = true;\n"
        "  return Status::Ok();\n"
        "}",
        6.5,
    )

    section_title(pdf, "2. Checkpoint okuma - checkpoint.cpp")
    code_block(
        pdf,
        "Result<TransformerModel> Checkpoint::Load(const std::string& path) {\n"
        "  std::ifstream input(path, std::ios::binary);\n"
        "  CheckpointHeader header{};\n"
        "  ReadBytes(input, &header, sizeof(header));\n"
        "  // magic + version dogrula\n"
        "  const ModelConfig config = HeaderToConfig(header);\n"
        "  TransformerModel model(config);\n"
        "  for (std::uint32_t tensorIndex = 0; tensorIndex < header.numTensors; ++tensorIndex) {\n"
        "    std::string name;\n"
        "    auto tensor = ReadTensor(input, name);\n"
        "    AssignTensor(model, name, tensor.Value());\n"
        "    // embedding, layer.0.attn..., lm_head\n"
        "  }\n"
        "  return Result<TransformerModel>::Ok(std::move(model));\n"
        "}",
        6.5,
    )

    section_title(pdf, "3. Prompt sablonu - dialogue_format.cpp")
    code_block(
        pdf,
        "std::string BuildInferencePrompt(const std::string& intent,\n"
        "                                 const std::string& userPrompt,\n"
        "                                 const std::string& systemPrompt) {\n"
        "  const std::string normalizedIntent =\n"
        "      intent.empty() ? nlp::DefaultIntent() : nlp::NormalizeIntentLabel(intent);\n"
        "  std::string sequence = \"Niyet: \" + normalizedIntent + \".\";\n"
        "  if (!systemPrompt.empty()) {\n"
        "    sequence += \" Sistem: \" + systemPrompt;\n"
        "  }\n"
        "  sequence += \" Kullanici: \" + userPrompt + \" Asistan:\";\n"
        "  return sequence;\n"
        "}\n"
        "\n"
        "std::string SanitizeGeneratedResponse(std::string text) {\n"
        "  const std::string marker = \"Asistan:\";\n"
        "  const std::size_t markerPos = text.find(marker);\n"
        "  if (markerPos != std::string::npos) {\n"
        "    text.erase(0, markerPos + marker.size());\n"
        "  }\n"
        "  const std::string intentMarker = \"Niyet:\";\n"
        "  const std::size_t intentPos = text.find(intentMarker);\n"
        "  if (intentPos != std::string::npos) {\n"
        "    text.erase(intentPos, text.size());\n"
        "  }\n"
        "  while (!text.empty() && (text.front() == ' ' || text.front() == '\\n'\n"
        "         || text.front() == '\\r')) {\n"
        "    text.erase(text.begin());\n"
        "  }\n"
        "  return text;\n"
        "}",
        6.5,
    )

    pdf.add_page()
    section_title(pdf, "4. Local backend - local_chat_backend.cpp")
    code_block(
        pdf,
        "ChatResponse LocalChatBackend::Complete(const ChatRequest& request) {\n"
        "  ChatResponse response;\n"
        "  if (!session_.IsLoaded()) {\n"
        "    response.error = \"local model is not loaded\";\n"
        "    return response;\n"
        "  }\n"
        "  if (request.prompt.empty()) {\n"
        "    response.error = \"prompt is empty\";\n"
        "    return response;\n"
        "  }\n"
        "  std::lock_guard<std::mutex> lock(mutex_);\n"
        "  inference::SamplingConfig& sampling = session_.Sampling();\n"
        "  const float savedTemperature = sampling.temperature;\n"
        "  const bool savedGreedy = sampling.greedy;\n"
        "  sampling.temperature = request.temperature;\n"
        "  sampling.greedy = request.greedy;\n"
        "  const std::string intent = request.intent.empty()\n"
        "      ? nlp::ClassifyIntent(request.prompt)\n"
        "      : nlp::NormalizeIntentLabel(request.intent);\n"
        "  const std::string modelPrompt =\n"
        "      training::BuildInferencePrompt(intent, request.prompt);\n"
        "  const auto result = session_.Complete(modelPrompt, request.maxTokens, rng_);\n"
        "  sampling.temperature = savedTemperature;\n"
        "  sampling.greedy = savedGreedy;\n"
        "  if (!result.IsOk()) {\n"
        "    response.error = result.GetError().message;\n"
        "    return response;\n"
        "  }\n"
        "  response.text = training::SanitizeGeneratedResponse(result.Value());\n"
        "  response.intent = intent;\n"
        "  return response;\n"
        "}",
        6.0,
    )

    section_title(pdf, "5. Oturum tamamlama - chat_session.cpp")
    code_block(
        pdf,
        "Result<std::string> ChatSession::Complete(const std::string& prompt,\n"
        "    const std::size_t maxNewTokens, std::mt19937& rng) {\n"
        "  if (!loaded_ || !generator_.has_value()) {\n"
        "    return Result<std::string>::Fail(ErrorCode::InvalidArgument,\n"
        "                                     \"chat session is not loaded\");\n"
        "  }\n"
        "  if (prompt.empty()) {\n"
        "    return Result<std::string>::Fail(ErrorCode::InvalidArgument,\n"
        "                                     \"prompt is empty\");\n"
        "  }\n"
        "  const std::vector<TokenId> promptIds =\n"
        "      tokenizer_.EncodeWithSpecialTokens(prompt, true, false);\n"
        "  const auto generated = generator_->Generate(promptIds, maxNewTokens, rng);\n"
        "  if (!generated.IsOk()) {\n"
        "    return Result<std::string>::Fail(generated.GetError().code,\n"
        "                                     generated.GetError().message);\n"
        "  }\n"
        "  const std::vector<TokenId>& allIds = generated.Value();\n"
        "  if (allIds.size() <= promptIds.size()) {\n"
        "    return Result<std::string>::Ok(std::string{});\n"
        "  }\n"
        "  const std::vector<TokenId> newIds(\n"
        "      allIds.begin() + static_cast<std::ptrdiff_t>(promptIds.size()), allIds.end());\n"
        "  return Result<std::string>::Ok(tokenizer_.Decode(newIds));\n"
        "}",
        6.0,
    )

    pdf.add_page()
    section_title(pdf, "6. Token uretim dongusu - generator.cpp")
    code_block(
        pdf,
        "Result<std::vector<TokenId>> TextGenerator::Generate(\n"
        "    const std::vector<TokenId>& prompt, const std::size_t maxNewTokens,\n"
        "    std::mt19937& rng) {\n"
        "  if (prompt.empty()) {\n"
        "    return Result<std::vector<TokenId>>::Fail(ErrorCode::InvalidArgument,\n"
        "        \"generate requires a non-empty prompt\");\n"
        "  }\n"
        "  if (maxNewTokens == 0) {\n"
        "    return Result<std::vector<TokenId>>::Ok(prompt);\n"
        "  }\n"
        "  const Dimension vocabSize = engine_.GetModelConfig().vocabSize;\n"
        "  std::vector<TokenId> sequence = prompt;\n"
        "  Tensor logits = Tensor::Zeros(Shape{1, vocabSize});\n"
        "  const Status prefillStatus = engine_.Prefill(prompt, logits);\n"
        "  if (!prefillStatus.IsOk()) {\n"
        "    return Result<std::vector<TokenId>>::Fail(prefillStatus.GetError().code,\n"
        "                                              prefillStatus.Message());\n"
        "  }\n"
        "  for (std::size_t step = 0; step < maxNewTokens; ++step) {\n"
        "    const TokenId nextToken = Sampler::Sample(logits, config_, rng);\n"
        "    sequence.push_back(nextToken);\n"
        "    if (nextToken == config_.eosTokenId) break;\n"
        "    if (engine_.SequenceLength() >= engine_.GetModelConfig().maxSeqLen) break;\n"
        "    const Status decodeStatus = engine_.Decode(nextToken, logits);\n"
        "    if (!decodeStatus.IsOk()) {\n"
        "      return Result<std::vector<TokenId>>::Fail(decodeStatus.GetError().code,\n"
        "                                                  decodeStatus.Message());\n"
        "    }\n"
        "  }\n"
        "  return Result<std::vector<TokenId>>::Ok(std::move(sequence));\n"
        "}",
        6.0,
    )

    section_title(pdf, "7. Forward pass - inference_engine.cpp")
    code_block(
        pdf,
        "Status InferenceEngine::Prefill(const std::vector<TokenId>& tokens,\n"
        "                                Tensor& logits) {\n"
        "  Reset();\n"
        "  Tensor hidden;\n"
        "  EmbedTokens(tokens, hidden);      // token ID -> vektor\n"
        "  Tensor output;\n"
        "  ForwardHidden(hidden, 0, output);   // 124 katman\n"
        "  Tensor lastRow = Tensor::Zeros(Shape{1, model_.GetConfig().hiddenDim});\n"
        "  const Dimension lastIndex = static_cast<Dimension>(tokens.size() - 1);\n"
        "  for (Dimension dim = 0; dim < model_.GetConfig().hiddenDim; ++dim) {\n"
        "    lastRow.At({0, static_cast<Index>(dim)}) =\n"
        "        output.At({static_cast<Index>(lastIndex), static_cast<Index>(dim)});\n"
        "  }\n"
        "  return LogitsFromRow(lastRow, logits); // norm + lm_head -> vocab skorlari\n"
        "}\n"
        "\n"
        "Status InferenceEngine::Decode(const TokenId token, Tensor& logits) {\n"
        "  Tensor hidden;\n"
        "  EmbedToken(token, hidden);\n"
        "  const std::size_t cacheStart = cache_.Length();\n"
        "  Tensor output;\n"
        "  ForwardHidden(hidden, cacheStart, output);\n"
        "  return LogitsFromRow(output, logits);\n"
        "}\n"
        "\n"
        "Status InferenceEngine::ForwardHidden(const Tensor& hidden,\n"
        "    const std::size_t cacheStart, Tensor& output) {\n"
        "  Tensor current = hidden;\n"
        "  for (std::size_t layer = 0; layer < model_.NumLayers(); ++layer) {\n"
        "    Tensor next = Tensor::Zeros(Shape{seqLen, model_.GetConfig().hiddenDim});\n"
        "    model_.Layer(layer).ForwardWithCache(current, model_.GetRopeCache(),\n"
        "                                         cache_, layer, cacheStart, next);\n"
        "    current = std::move(next);\n"
        "  }\n"
        "  output = std::move(current);\n"
        "  return Status::Ok();\n"
        "}",
        5.8,
    )

    pdf.add_page()
    section_title(pdf, "8. Kelime secimi - sampler.cpp")
    code_block(
        pdf,
        "TokenId Sampler::Sample(const Tensor& logits, const SamplingConfig& config,\n"
        "                        std::mt19937& rng) {\n"
        "  if (config.greedy || config.temperature <= 0.0f) {\n"
        "    return SampleGreedy(logits);  // en yuksek skor\n"
        "  }\n"
        "  std::vector<Scalar> row = LogitsRow(logits);\n"
        "  ApplyTemperature(row, config.temperature);\n"
        "  ApplyTopK(row, config.topK);\n"
        "  SoftmaxInPlace(row);\n"
        "  ApplyTopP(row, config.topP);\n"
        "  return SampleFromDistribution(row, rng);\n"
        "}",
        6.5,
    )

    section_title(pdf, "Kisa Ozet")
    pdf.set_font("ArialB", "", 10)
    table_row(pdf, "Ne", "Nerede")
    pdf.set_font("Arial", "", 9)
    table_row(pdf, "Dosya -> bellek", "Checkpoint::Load / QuantCheckpoint::LoadRuntime")
    table_row(pdf, "Metin -> sayi", "BpeTokenizer::Encode")
    table_row(pdf, "Sayi -> anlam vektoru", "EmbedTokens + kayitli agirliklar")
    table_row(pdf, "Siradaki kelime?", "ForwardHidden -> logits")
    table_row(pdf, "Kelime sec", "Sampler::Sample")
    table_row(pdf, "Sayi -> metin", "BpeTokenizer::Decode")
    table_row(pdf, "HTTP", "ChatHttpServer POST /api/v1/chat")
    pdf.ln(4)

    body(
        pdf,
        "Ogrenme egitimde olur; inference sadece kayitli agirliklarla 'bir sonraki token' "
        "tahminini tekrarlar. Model bir veritabani degildir; her cevap o anda hesaplanir.",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    try:
        path = build_pdf()
        print(f"OK: {path}")
    except Exception as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
