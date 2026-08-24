export class ChatApiException extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ChatApiException';
  }
}

export interface HealthInfo {
  loaded: boolean;
  stage: string;
  backend?: string;
  model?: string;
}

export interface ServerConfig {
  backend: string;
  model: string;
  localAvailable: boolean;
  openAiAvailable: boolean;
  openaiModel: string;
}

export interface ChatCompletion {
  text: string;
  intent?: string;
}

export class ChatApi {
  constructor(private readonly baseUrl: string) {}

  private uri(path: string): string {
    const normalized = this.baseUrl.endsWith('/')
      ? this.baseUrl.slice(0, -1)
      : this.baseUrl;
    return `${normalized}${path}`;
  }

  async health(): Promise<HealthInfo> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(this.uri('/api/v1/health'), {
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ChatApiException(`health status ${response.status}`);
      }

      const json = (await response.json()) as Record<string, unknown>;
      return {
        loaded: json.loaded === true,
        stage: json.stage?.toString() ?? '',
        backend: json.backend?.toString(),
        model: json.model?.toString(),
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async getConfig(): Promise<ServerConfig> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(this.uri('/api/v1/config'), {
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ChatApiException(`config status ${response.status}`);
      }

      const json = (await response.json()) as Record<string, unknown>;
      return {
        backend: json.backend?.toString() ?? 'openai',
        model: json.model?.toString() ?? '',
        localAvailable: json.local_available === true,
        openAiAvailable: json.openai_available === true,
        openaiModel: json.openai_model?.toString() ?? 'llama3.2',
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async setConfig(params: {
    backend: 'local' | 'openai';
    openaiModel: string;
  }): Promise<ServerConfig> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    try {
      const response = await fetch(this.uri('/api/v1/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backend: params.backend,
          openai_model: params.openaiModel,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const json = (await response.json()) as Record<string, unknown>;
        throw new ChatApiException(json.error?.toString() ?? `config status ${response.status}`);
      }

      const json = (await response.json()) as Record<string, unknown>;
      return {
        backend: json.backend?.toString() ?? params.backend,
        model: json.model?.toString() ?? '',
        localAvailable: json.local_available === true,
        openAiAvailable: json.openai_available === true,
        openaiModel: json.openai_model?.toString() ?? params.openaiModel,
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async complete(params: {
    prompt: string;
    maxTokens: number;
    temperature: number;
    greedy: boolean;
  }): Promise<ChatCompletion> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120_000);

    try {
      const response = await fetch(this.uri('/api/v1/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: params.prompt,
          max_tokens: params.maxTokens,
          temperature: params.temperature,
          greedy: params.greedy,
        }),
        signal: controller.signal,
      });

      if (response.status !== 200 && response.status !== 400) {
        throw new ChatApiException(`chat status ${response.status}`);
      }

      const json = (await response.json()) as Record<string, unknown>;
      const error = json.error;
      if (error != null && String(error).length > 0) {
        throw new ChatApiException(String(error));
      }

      return {
        text: json.text?.toString() ?? '',
        intent: json.intent?.toString(),
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  async reset(): Promise<void> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    try {
      const response = await fetch(this.uri('/api/v1/reset'), {
        method: 'POST',
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ChatApiException(`reset status ${response.status}`);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
}
