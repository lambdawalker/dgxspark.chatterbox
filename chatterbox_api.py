"""HTTP API for Chatterbox. Run: uv run uvicorn chatterbox_api:app --host 0.0.0.0 --port 8000

Single-worker deployment. Jobs and event history are persisted in SQLite; interrupted
jobs are marked failed on restart. SSE supports Last-Event-ID reconnection.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import sqlite3
import tempfile
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("CHATTERBOX_API_DATA", "./.chatterbox-api")).resolve()
DB = ROOT / "jobs.sqlite3"
MODELS = ("chatterbox", "chatterbox-turbo", "chatterbox-nano", "chatterbox-multilingual", "chatterbox-vc")
MAX_UPLOAD = int(os.getenv("CHATTERBOX_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_JOBS = int(os.getenv("CHATTERBOX_MAX_QUEUED_JOBS", "30"))
DEVICE = os.getenv("CHATTERBOX_DEVICE", "auto")
job_queue: queue.Queue[str] = queue.Queue(maxsize=MAX_JOBS)
models: dict[str, object] = {}
model_lock = threading.Lock()
db_lock = threading.RLock()
stop = threading.Event()


def db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init():
    ROOT.mkdir(parents=True, exist_ok=True)
    with db_lock, db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, state TEXT NOT NULL, stage TEXT NOT NULL,
          progress INTEGER NOT NULL, request TEXT NOT NULL, result TEXT,
          error TEXT, created REAL NOT NULL, updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
          payload TEXT NOT NULL, created REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS events_job_id ON events(job_id,id);
        """)
        # An inference interrupted by a process restart cannot be resumed.
        rows = c.execute("SELECT id FROM jobs WHERE state IN ('queued','running')").fetchall()
        for row in rows:
            change(row["id"], "failed", "interrupted", 100, error="Server restarted during processing", connection=c)


def change(jid, state, stage, progress, *, result=None, error=None, connection=None):
    payload = {"id": jid, "state": state, "stage": stage, "progress": progress,
               "result_url": f"/v1/jobs/{jid}/audio" if result else None, "error": error}
    with db_lock:
        if connection is None:
            with db() as c:
                change(jid, state, stage, progress, result=result, error=error, connection=c)
            return
        connection.execute(
            "UPDATE jobs SET state=?,stage=?,progress=?,result=COALESCE(?,result),error=?,updated=? WHERE id=?",
            (state, stage, progress, result, error, time.time(), jid))
        connection.execute("INSERT INTO events(job_id,payload,created) VALUES(?,?,?)",
                           (jid, json.dumps(payload), time.time()))


def snapshot(jid):
    with db_lock, db() as c:
        row = c.execute("SELECT id,state,stage,progress,error,result,created,updated FROM jobs WHERE id=?", (jid,)).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    result = dict(row)
    result["result_url"] = f"/v1/jobs/{jid}/audio" if result.pop("result") else None
    return result


def load_model(name):
    with model_lock:
        if name in models:
            return models[name]
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
        if name == "chatterbox":
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device)
        elif name in ("chatterbox-turbo", "chatterbox-nano"):
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device, nano=name.endswith("nano"))
        elif name == "chatterbox-multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device)
        else:
            from chatterbox.vc import ChatterboxVC
            model = ChatterboxVC.from_pretrained(device)
        models[name] = model
        return model


def save_wav(output, sample_rate, tensor):
    import numpy as np
    array = tensor.squeeze().detach().cpu().numpy()
    array = (np.clip(array, -1, 1) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(array.tobytes())


def generate(jid, spec):
    model_name = spec["model"]
    change(jid, "running", "loading_model", 10)
    model = load_model(model_name)
    change(jid, "running", "generating", 35)
    params = spec["parameters"]
    reference = spec.get("reference")
    if model_name == "chatterbox-vc":
        wav = model.generate(audio=spec["source"], target_voice_path=reference)
    else:
        kwargs = dict(params)
        kwargs["audio_prompt_path"] = reference
        if model_name == "chatterbox-multilingual":
            kwargs["language_id"] = spec.get("language_id") or "en"
        # A shared model caches voice conditionals. Always require an explicit
        # reference for reproducibility and to prevent another job's voice leaking.
        wav = model.generate(spec["input"], **kwargs)
    change(jid, "running", "encoding", 90)
    outfile = ROOT / f"{jid}.wav"
    save_wav(outfile, model.sr, wav)
    change(jid, "completed", "completed", 100, result=str(outfile))


def worker():
    while not stop.is_set():
        try:
            jid = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            with db_lock, db() as c:
                row = c.execute("SELECT request,state FROM jobs WHERE id=?", (jid,)).fetchone()
            if row and row["state"] == "queued":
                generate(jid, json.loads(row["request"]))
        except Exception as exc:
            change(jid, "failed", "failed", 100, error=f"{type(exc).__name__}: {exc}")
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app):
    init()
    stop.clear()
    thread = threading.Thread(target=worker, daemon=True, name="chatterbox-inference")
    thread.start()
    yield
    stop.set()
    thread.join(timeout=2)


app = FastAPI(title="Chatterbox HTTP API", version="0.1.0", lifespan=lifespan)


def authorize(authorization: str | None = Header(default=None)):
    token = os.environ.get("CHATTERBOX_API_KEY")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(401, "Invalid API key")


class TTSRequest(BaseModel):
    model: str = "chatterbox-turbo"
    input: str = Field(min_length=1, max_length=4096)
    language_id: str | None = None
    reference_id: str | None = None
    temperature: float = Field(0.8, gt=0, le=5)
    top_p: float = Field(0.95, ge=0, le=1)
    repetition_penalty: float = Field(1.2, ge=1, le=2)
    top_k: int = Field(1000, ge=0, le=1000)
    min_p: float = Field(0, ge=0, le=1)
    exaggeration: float = Field(0.5, ge=0, le=2)
    cfg_weight: float = Field(0.5, ge=0, le=1)
    norm_loudness: bool = True


def validate(req):
    if req.model not in MODELS or req.model == "chatterbox-vc":
        raise HTTPException(422, "Choose a supported TTS model")
    if req.model in ("chatterbox-turbo", "chatterbox-nano"):
        if any(k in req.model_fields_set and getattr(req, k) != 0 for k in ("cfg_weight", "exaggeration", "min_p")):
            raise HTTPException(422, "Turbo/Nano do not support cfg_weight, exaggeration, or min_p; use 0")
    elif ("top_k" in req.model_fields_set or "norm_loudness" in req.model_fields_set):
        raise HTTPException(422, "top_k and norm_loudness are Turbo/Nano-only parameters")
    if req.model != "chatterbox-multilingual" and req.language_id:
        raise HTTPException(422, "language_id is available only for multilingual TTS")


def voice_path(voice_id):
    if voice_id is None:
        raise HTTPException(422, "reference_id is required; register a reference voice first")
    path = ROOT / "voices" / f"{voice_id}.wav"
    try:
        valid = str(uuid.UUID(voice_id)) == voice_id
    except ValueError:
        valid = False
    if not path.is_file() or not valid:
        raise HTTPException(404, "Voice not found")
    return str(path)


def submit(spec):
    jid = str(uuid.uuid4())
    now = time.time()
    with db_lock, db() as c:
        c.execute("INSERT INTO jobs(id,state,stage,progress,request,created,updated) VALUES(?,?,?,?,?,?,?)",
                  (jid, "queued", "queued", 0, json.dumps(spec), now, now))
        change(jid, "queued", "queued", 0, connection=c)
    try:
        job_queue.put_nowait(jid)
    except queue.Full:
        change(jid, "failed", "queue_full", 100, error="Queue full")
        raise HTTPException(503, "Inference queue full")
    return {"id": jid, "status_url": f"/v1/jobs/{jid}",
            "events_url": f"/v1/jobs/{jid}/events", "audio_url": f"/v1/jobs/{jid}/audio"}


@app.get("/health")
def health():
    return {"status": "ok", "queue_depth": job_queue.qsize()}


@app.get("/v1/models", dependencies=[Depends(authorize)])
def list_models():
    return {"data": [{"id": name, "loaded": name in models} for name in MODELS]}


@app.post("/v1/voices", dependencies=[Depends(authorize)])
async def create_voice(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(415, "Only WAV reference audio is supported")
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Audio too large")
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            if reader.getnframes() == 0:
                raise ValueError("Empty audio")
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(422, f"Invalid WAV: {exc}") from exc
    ident = str(uuid.uuid4())
    folder = ROOT / "voices"
    folder.mkdir(exist_ok=True)
    (folder / f"{ident}.wav").write_bytes(data)
    return {"id": ident}


@app.get("/v1/voices", dependencies=[Depends(authorize)])
def list_voices():
    folder = ROOT / "voices"
    return {"data": [{"id": p.stem} for p in folder.glob("*.wav")] if folder.exists() else []}


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(authorize)])
def delete_voice(voice_id: uuid.UUID):
    path = ROOT / "voices" / f"{voice_id}.wav"
    if not path.exists():
        raise HTTPException(404, "Voice not found")
    path.unlink()
    return {"deleted": True}


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
def create_tts_job(req: TTSRequest):
    validate(req)
    params = req.model_dump(exclude={"input", "model", "language_id", "reference_id"})
    if req.model not in ("chatterbox-turbo", "chatterbox-nano"):
        params.pop("top_k")
        params.pop("norm_loudness")
    else:
        params.pop("exaggeration")
        params.pop("cfg_weight")
        params.pop("min_p")
    return submit({"model": req.model, "input": req.input, "language_id": req.language_id,
                   "reference": voice_path(req.reference_id), "parameters": params})


@app.post("/v1/voice-conversions", status_code=202, dependencies=[Depends(authorize)])
async def create_vc_job(source: UploadFile = File(...), reference_id: str = Form(...)):
    if source.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(415, "Only WAV input is supported")
    data = await source.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Audio too large")
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            if reader.getnframes() == 0:
                raise ValueError("Empty audio")
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(422, f"Invalid WAV: {exc}") from exc
    source_id = str(uuid.uuid4())
    source_path = ROOT / f"{source_id}-input.wav"
    reference = voice_path(reference_id)
    source_path.write_bytes(data)
    return submit({"model": "chatterbox-vc", "source": str(source_path),
                   "reference": reference, "parameters": {}})


@app.get("/v1/jobs/{jid}", dependencies=[Depends(authorize)])
def job_status(jid: uuid.UUID):
    return snapshot(str(jid))


@app.get("/v1/jobs/{jid}/audio", dependencies=[Depends(authorize)])
def job_audio(jid: uuid.UUID):
    row = snapshot(str(jid))
    if row["state"] != "completed":
        raise HTTPException(409, "Audio is not ready")
    path = ROOT / f"{jid}.wav"
    if not path.exists():
        raise HTTPException(410, "Audio expired")
    return FileResponse(path, media_type="audio/wav", filename=f"{jid}.wav")


@app.get("/v1/jobs/{jid}/events", dependencies=[Depends(authorize)])
async def job_events(jid: uuid.UUID, request: Request, last_event_id: str | None = Header(None)):
    jid = str(jid)
    snapshot(jid)
    try:
        cursor = max(0, int(last_event_id or request.query_params.get("after", "0")))
    except ValueError:
        raise HTTPException(422, "Invalid event cursor")

    async def stream():
        nonlocal cursor
        while True:
            if await request.is_disconnected():
                break
            with db_lock, db() as c:
                rows = c.execute("SELECT id,payload FROM events WHERE job_id=? AND id>? ORDER BY id",
                                 (jid, cursor)).fetchall()
                current = c.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()
            for row in rows:
                cursor = row["id"]
                yield f"id: {cursor}\nevent: progress\ndata: {row['payload']}\n\n"
            if current["state"] in ("completed", "failed"):
                break
            if not rows:
                yield ": keep-alive\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
