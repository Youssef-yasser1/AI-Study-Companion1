# ============================================
# run_public.py
# Launch the AI Study Companion with a PUBLIC URL
# Uses Cloudflare tunnel - free, no signup needed!
# ============================================

import subprocess
import threading
import time
import re
import sys
import os

# ============================================
# Configuration
# ============================================
PORT = 5000
CLOUDFLARED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")


def start_tunnel():
    """Start Cloudflare tunnel and print the public URL."""
    time.sleep(3)  # Wait for Flask to start
    
    print("\n" + "=" * 60)
    print("  [*] Starting Cloudflare tunnel...")
    print("  [*] Please wait a few seconds...")
    print("=" * 60, flush=True)
    
    try:
        proc = subprocess.Popen(
            [CLOUDFLARED_PATH, "tunnel", "--url", f"http://localhost:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        # cloudflared outputs the URL to stderr
        url_found = False
        
        def read_stream(stream, label):
            nonlocal url_found
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                # Look for the trycloudflare.com URL
                if "trycloudflare.com" in line and not url_found:
                    url_match = re.search(r'(https?://[a-zA-Z0-9\-]+\.trycloudflare\.com)', line)
                    if url_match:
                        public_url = url_match.group(1)
                        url_found = True
                        print("\n" + "=" * 60)
                        print("  *** PUBLIC LINK READY! ***")
                        print("=" * 60)
                        print(f"\n  -> Public URL: {public_url}")
                        print(f"  -> Local URL:  http://127.0.0.1:{PORT}")
                        print(f"\n  => Share the Public URL with anyone!")
                        print("  => They can open it in any browser.")
                        print("  => Keep this terminal open while sharing.")
                        print("=" * 60 + "\n", flush=True)
        
        # Read both stdout and stderr in separate threads
        t1 = threading.Thread(target=read_stream, args=(proc.stdout, "OUT"), daemon=True)
        t2 = threading.Thread(target=read_stream, args=(proc.stderr, "ERR"), daemon=True)
        t1.start()
        t2.start()
        
        # Wait for both to finish or URL to be found
        t1.join()
        t2.join()
                    
    except FileNotFoundError:
        print(f"\n  [ERROR] cloudflared.exe not found at: {CLOUDFLARED_PATH}")
        print("  [INFO] Download it from: https://github.com/cloudflare/cloudflared/releases", flush=True)
    except Exception as e:
        print(f"\n  [ERROR] Tunnel error: {e}", flush=True)


if __name__ == '__main__':
    # Check cloudflared exists
    if not os.path.exists(CLOUDFLARED_PATH):
        print(f"[ERROR] cloudflared.exe not found!")
        print(f"Expected at: {CLOUDFLARED_PATH}")
        print("Download from: https://github.com/cloudflare/cloudflared/releases")
        sys.exit(1)
    
    # Start tunnel in a separate thread
    threading.Thread(target=start_tunnel, daemon=True).start()
    
    # Import and run the Flask app
    from app import app
    
    print("\n" + "=" * 50)
    print("  AI Study Companion - Public Server")
    print(f"  Local:  http://127.0.0.1:{PORT}")
    print("  Waiting for public tunnel link...")
    print("=" * 50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=False
    )
