<?php

namespace Laravel\Ai\Providers;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Str;
use Laravel\Ai\Ai;
use Laravel\Ai\Contracts\Providers\AudioProvider;
use Laravel\Ai\Contracts\Providers\EmbeddingProvider;
use Laravel\Ai\Contracts\Providers\TextProvider;
use Laravel\Ai\Events\AudioGenerated;
use Laravel\Ai\Events\GeneratingAudio;
use Laravel\Ai\Prompts\AudioPrompt;
use Laravel\Ai\Responses\AudioResponse;
use Laravel\Ai\Responses\Data\Meta;

class AzureOpenAiProvider extends Provider implements AudioProvider, EmbeddingProvider, TextProvider
{
    use Concerns\GeneratesEmbeddings;
    use Concerns\GeneratesText;
    use Concerns\HasAudioGateway;
    use Concerns\HasEmbeddingGateway;
    use Concerns\HasTextGateway; 
    use Concerns\StreamsText;

    /**
     * Get the credentials for the AI provider.
     * 
     * Azure OpenAI uses API key authentication via the `api-key` header.
     */
    public function providerCredentials(): array
    {
        return [
            'key' => $this->config['key'],
        ];
    }

    /**
     * Get the provider connection configuration other than the driver, key, and name.
     */
    public function additionalConfiguration(): array
    {
        return array_filter([
            'url' => $this->buildAzureBaseUrl(),
            'api_version' => $this->config['api_version'] ?? '2024-10-21',
        ]);
    }

    /**
     * Build the Azure OpenAI base URL for text/embeddings.
     */
    protected function buildAzureBaseUrl(): string
    {
        $endpoint = rtrim($this->config['endpoint'] ?? '', '/');
        
        // Use OpenAI-compatible endpoint format
        return "{$endpoint}/openai/v1";
    }

    /**
     * Generate audio from the given text using Azure's TTS endpoint.
     * 
     * Azure TTS requires a deployment-based URL format:
     * https://{resource}.openai.azure.com/openai/deployments/{deployment}/audio/speech?api-version=xxx
     */
    public function audio(
        string $text,
        string $voice = 'alloy',
        ?string $instructions = null,
        ?string $model = null,
    ): AudioResponse {
        $invocationId = (string) Str::uuid7();
        $model ??= $this->defaultAudioModel();

        $prompt = new AudioPrompt($text, $voice, $instructions, $this, $model);

        if (Ai::audioIsFaked()) {
            Ai::recordAudioGeneration($prompt);
        }

        $this->events->dispatch(new GeneratingAudio(
            $invocationId, $this, $model, $prompt,
        ));

        // Build Azure TTS endpoint URL
        $endpoint = rtrim($this->config['endpoint'] ?? '', '/');
        $apiVersion = $this->config['api_version'] ?? '2024-10-21';
        $url = "{$endpoint}/openai/deployments/{$model}/audio/speech?api-version={$apiVersion}";

        // Make direct HTTP request to Azure TTS
        $response = Http::withToken($this->config['key'])
            ->post($url, array_filter([
                'model' => $model,
                'input' => $text,
                'voice' => $voice,
                'instructions' => $instructions,
                'response_format' => 'mp3',
            ]));

        if (!$response->successful()) {
            throw new \Exception("Azure TTS Error [{$response->status()}]: " . $response->body());
        }

        $audioContent = $response->body();
        $base64Audio = base64_encode($audioContent);

        $audioResponse = new AudioResponse(
            audio: $base64Audio,
            meta: new Meta(
                provider: $this->name(),
                model: $model,
            ),
            mime: 'audio/mpeg',
        );

        $this->events->dispatch(new AudioGenerated(
            $invocationId, $this, $model, $prompt, $audioResponse,
        ));

        return $audioResponse;
    }

    /**
     * Get the name of the default (deployment name) text model.
     */
    public function defaultTextModel(): string
    {
        return $this->config['deployment'] ?? 'gpt-4o';
    }

    /**
     * Get the name of the cheapest text model.
     */
    public function cheapestTextModel(): string
    {
        return $this->config['deployment'] ?? 'gpt-4o-mini';
    }

    /**
     * Get the name of the smartest text model.
     */
    public function smartestTextModel(): string
    {
        return $this->config['deployment'] ?? 'gpt-4o';
    }

    /**
     * Get the name of the default embeddings model.
     */
    public function defaultEmbeddingsModel(): string
    {
        return $this->config['embedding_deployment'] ?? 'text-embedding-3-small';
    }

    /**
     * Get the default dimensions of the default embeddings model.
     */
    public function defaultEmbeddingsDimensions(): int
    {
        return 1536;
    }

    /**
     * Get the name of the default audio (TTS) model.
     */
    public function defaultAudioModel(): string
    {
        return $this->config['audio_deployment'] ?? 'gpt-4o-mini-tts';
    }
}

