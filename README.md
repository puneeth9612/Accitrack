# AcciTrack - Driver Monitoring (patched)

**Updated:** 2025-11-09 10:21:54

## Features
- Drowsiness & yawning via MediaPipe FaceMesh (EAR/MAR).
- Distraction via YOLOv8n on COCO classes (e.g., phone/food/remote/book).
- Audible alerts (pygame), optional SMS (Twilio/Fast2SMS), optional Supabase logging.
- Cooldowns to avoid spamming (sound vs notifications).
- Robust init + clean shutdown; safe when some deps/creds are missing.

## Quickstart
1. Create and activate a Python 3.10–3.12 environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill values (optional for SMS/Supabase).
4. Place `alert.mp3` in this folder (or set `ALERT_SOUND` in `.env`).
5. Run:
   ```bash
   python driver_monitor_fixed.py --show-metrics
   ```
   - Press `q` to quit.
   - Use `--test-sms` to send a single test SMS (if you enable SMS in code).

## Notes
- To enable SMS, set `ENABLE_SMS = True` in `driver_monitor_fixed.py` and provide credentials in `.env`.
- On Windows, if you see camera backend issues, try unplug/replug or change indices in `open_camera()`.
- If MediaPipe wheels are not available for your Python version, switch to 3.10–3.12.

## Safety
- Notifications are throttled (`ALERT_NOTIFY_MIN_INTERVAL_S`) per alert type.
- Sound is throttled (`ALERT_SOUND_MIN_INTERVAL_S`) to avoid constant noise.
- Supabase insertions are best-effort and non-blocking.

## License
Educational use.
