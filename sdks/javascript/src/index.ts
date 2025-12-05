/**
 * Sentinella JavaScript/TypeScript SDK
 */

import axios, { AxiosInstance } from 'axios';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatCompletionOptions {
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  session_id?: string;
}

export interface ChatCompletionResponse {
  id: string;
  model: string;
  choices: Array<{
    message: ChatMessage;
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  latency_ms: number;
  cached: boolean;
}

export class SentinellaClient {
  private client: AxiosInstance;
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl: string = 'http://localhost:8000') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      headers: {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
      },
      timeout: 60000,
    });
  }

  async chat(
    messages: ChatMessage[],
    options: ChatCompletionOptions = {}
  ): Promise<ChatCompletionResponse> {
    const response = await this.client.post('/v1/chat/completions', {
      messages,
      ...options,
    });
    return response.data;
  }

  async *chatStream(
    messages: ChatMessage[],
    options: ChatCompletionOptions = {}
  ): AsyncGenerator<string, void, unknown> {
    const response = await this.client.post(
      '/v1/chat/completions',
      {
        messages,
        ...options,
        stream: true,
      },
      {
        responseType: 'stream',
      }
    );

    for await (const chunk of response.data) {
      const lines = chunk.toString().split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.choices?.[0]?.delta?.content) {
              yield parsed.choices[0].delta.content;
            }
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    }
  }

  async listModels(): Promise<Array<{ id: string; owned_by: string }>> {
    const response = await this.client.get('/v1/models');
    return response.data.data || [];
  }

  async getMetrics(): Promise<any> {
    const response = await this.client.get('/metrics');
    return response.data;
  }

  async renderTemplate(
    templateId: string,
    variables: Record<string, any>
  ): Promise<string> {
    const response = await this.client.post(
      `/v1/templates/${templateId}/render`,
      { variables }
    );
    return response.data.rendered;
  }
}

export default SentinellaClient;

