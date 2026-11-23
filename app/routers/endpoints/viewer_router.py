"""
Live viewer page — uses MJPEG for smooth real-time video.
No JavaScript refresh flicker — the browser streams frames natively
just like a webcam feed.
"""
from urllib.parse import quote

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.security import verify_api_key_or_query

viewer_router = APIRouter()


@viewer_router.get("/view/{camera_id}", response_class=HTMLResponse, tags=["Live Viewer"])
def live_viewer(camera_id: int, key: str = Depends(verify_api_key_or_query)):
    """
    Open this in your browser to watch smooth live annotated video.
    Uses MJPEG — the browser streams frames continuously like a real video.
    """
    # url-safe, so it is also safe inside the HTML/JS below; empty while auth is off
    qs = "?key=" + quote(key, safe="") if key else ""
    qs_amp = "&" + qs[1:] if qs else ""
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Live Camera {camera_id}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d0d0d;
      color: #fff;
      font-family: 'Segoe UI', Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
      min-height: 100vh;
    }}
    h2 {{
      color: #4CAF50;
      font-size: 20px;
      margin-bottom: 16px;
      letter-spacing: 1px;
    }}
    .video-wrap {{
      position: relative;
      width: 100%;
      max-width: 900px;
      background: #111;
      border: 2px solid #4CAF50;
      border-radius: 8px;
      overflow: hidden;
      min-height: 240px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    #stream {{
      width: 100%;
      display: block;
    }}
    #overlay {{
      position: absolute;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 10px;
      color: #ffcc00;
      font-size: 16px;
    }}
    .spinner {{
      width: 42px; height: 42px;
      border: 4px solid #333;
      border-top-color: #4CAF50;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .info-bar {{
      margin-top: 12px;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      justify-content: center;
      font-size: 13px;
      color: #aaa;
    }}
    .legend {{
      display: flex;
      gap: 20px;
      margin-top: 10px;
      font-size: 13px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .dot {{
      width: 12px; height: 12px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 5px;
      vertical-align: middle;
    }}
    #status-pill {{
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 12px;
      color: #ffcc00;
    }}
  </style>
</head>
<body>
  <h2>🎥 Live Camera {camera_id} — Access Control View</h2>

  <div class="video-wrap">
    <img id="stream"
         src="/api/v1/stream/video/{camera_id}{qs}"
         alt=""
         onload="document.getElementById('overlay').style.display='none';"
         onerror="showWaiting();"/>
    <div id="overlay">
      <div class="spinner"></div>
      <div id="wait-msg">Loading stream — YOLO model starting up…</div>
      <div id="status-pill">checking...</div>
    </div>
  </div>

  <div class="info-bar">
    <span>Camera ID: <strong>{camera_id}</strong></span>
    <span id="ts">—</span>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:#00c800"></span>ALLOWED</span>
    <span><span class="dot" style="background:#dc0000"></span>UNAUTHORIZED / UNKNOWN</span>
    <span><span class="dot" style="background:#ffa500"></span>VERIFY</span>
    <span><span class="dot" style="background:#888"></span>PROCESSING</span>
  </div>

  <script>
    var shown = false;

    function showWaiting() {{
      if (shown) return;
      document.getElementById('overlay').style.display = 'flex';
      pollStatus();
    }}

    function pollStatus() {{
      fetch('/api/v1/stream/debug/{camera_id}{qs}')
      .then(r => r.json())
      .then(d => {{
        var pill = document.getElementById('status-pill');
        pill.textContent = 'Status: ' + d.status;
        if (d.error) {{
          pill.textContent += ' | ' + d.error;
          pill.style.color = '#ff5555';
        }}
        // If running but stream img errored, reload it to retry
        if (d.status === 'running' && !shown) {{
          setTimeout(reloadStream, 2000);
        }} else if (d.status !== 'running') {{
          setTimeout(pollStatus, 2000);
        }}
      }})
      .catch(() => setTimeout(pollStatus, 3000));
    }}

    function reloadStream() {{
      var img = document.getElementById('stream');
      img.src = '/api/v1/stream/video/{camera_id}?t=' + Date.now() + '{qs_amp}';
    }}

    // Update clock
    setInterval(function() {{
      document.getElementById('ts').textContent = new Date().toLocaleTimeString();
    }}, 1000);

    // Hide overlay once image loads
    document.getElementById('stream').addEventListener('load', function() {{
      shown = true;
      document.getElementById('overlay').style.display = 'none';
    }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
