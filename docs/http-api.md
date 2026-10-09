# Chatterbox HTTP API

The API runs separately from the existing Gradio demos. It exposes standard, Turbo,
Nano and multilingual TTS plus voice conversion. Model inference is serialized in one
background worker; reference audio is registered as a reusable voice. The model weights
are loaded lazily on the first request.

## Installation and start

```bash
uv sync --extra api
export CHATTERBOX_API_KEY="replace-with-a-random-secret"
uv run uvicorn chatterbox_api:app --host 127.0.0.1 --port 8000 --workers 1
```

For LAN access, bind to `0.0.0.0` and restrict incoming traffic with a firewall or
reverse proxy using HTTPS. Avoid exposing the API without `CHATTERBOX_API_KEY`.
The server uses a single in-process queue: **do not start multiple Uvicorn workers**
against the same data directory. Set `CHATTERBOX_API_DATA` for persistent files,
`CHATTERBOX_DEVICE` (default auto), `CHATTERBOX_MAX_UPLOAD_BYTES`, and
`CHATTERBOX_MAX_QUEUED_JOBS` as needed. The output is WAV.

## Register a reference voice

```bash
curl -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  -F "file=@reference.wav" http://localhost:8000/v1/voices
# {"id":"..."}
```

Register a WAV file containing a voice you have permission to use. Turbo/Nano
require a reference longer than five seconds. Registration does not precompute
model voice embeddings. Uploaded WAVs are not automatically deleted.

## Generate speech asynchronously

```bash
curl -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello world!","reference_id":"VOICE_UUID"}' \
  http://localhost:8000/v1/jobs
```

A 202 response supplies a job ID and status/events/audio URLs.

```bash
curl -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  http://localhost:8000/v1/jobs/JOB_UUID
curl -N -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  http://localhost:8000/v1/jobs/JOB_UUID/events
curl -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  http://localhost:8000/v1/jobs/JOB_UUID/audio -o output.wav
```

The SSE endpoint includes monotonically increasing event IDs and supports the
`Last-Event-ID` header or `?after=EVENT_ID`. Event history survives server
restarts, so clients can reconnect without losing completed events. Poll the status
endpoint as a fallback. Stages are queued, loading_model, generating, encoding,
completed or failed. **Percentages are coarse lifecycle markers**, not granular
neural inference progress. Disconnecting from SSE never cancels the generation.

Voice conversion requires an input WAV:

```bash
curl -H "Authorization: Bearer $CHATTERBOX_API_KEY" \
  -F "source=@input.wav" -F "reference_id=VOICE_UUID" \
  http://localhost:8000/v1/voice-conversions
```

`GET /health`, `GET /v1/models`, `GET /v1/voices`, and
`DELETE /v1/voices/{id}` are also available. Explore the OpenAPI schema at
`/docs`.

## Model-specific settings

All TTS models: `temperature`, `top_p`, `repetition_penalty`, with
`reference_id` required. Standard / multilingual:
`min_p`, `exaggeration`, `cfg_weight`. Turbo / Nano: `top_k`,
`norm_loudness`. Unsupported explicitly supplied controls return HTTP 422
rather than being silently ignored. `language_id` applies only to
`chatterbox-multilingual`.

## Operational limits and recovery

Jobs, events, outputs, and uploaded references persist under `CHATTERBOX_API_DATA`.
After a process restart, previously completed audio remains available and queued /
running jobs are marked failed. They **are not automatically retried**. For this
single-process implementation, deploy exactly one worker. Job cancellation, automatic
retention/cleanup, distributed queueing, and truly incremental model-audio streaming
are not implemented. Add these before exposing the server as a multi-tenant public
service. SSE is a notification channel, not an uninterrupted inference transport.

The API is **not yet OpenAI-compatible**: it uses durable job endpoints rather than
`POST /v1/audio/speech`.
