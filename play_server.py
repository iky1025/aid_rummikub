"""Local web GUI for playing against / spectating the distilled student.

No third-party deps — stdlib http.server only. Run:

    python play_server.py --model distill_s1s_dagger1.pt --port 8000

then open http://localhost:8000 in a browser.

Single-session (one local player). Modes:
  - versus   : you play one seat, the model the other; every model turn shows
               its candidate ranking (logits), whether it matched the greedy
               max-play, its predicted picture of your hidden hand (aux head),
               and what it would play in your shoes.
  - spectate : model vs greedy-ILP, step through ply by ply.
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from play_core import Student, GameSession, MAX_CANDIDATES

HTML_PATH = os.path.join(os.path.dirname(__file__), "play.html")

_lock = threading.Lock()
_state = {"student": None, "game": None, "spectate": False, "args": None}


def _new_game(mode, seed, meld, first):
    student = _state["student"]
    spectate = (mode == "spectate")
    if spectate:
        model_seat, human_seat = 0, 1
    else:
        # whoever is "first" takes seat 0 (seat 0 always moves first)
        if first == "model":
            model_seat, human_seat = 0, 1
        else:
            model_seat, human_seat = 1, 0
    g = GameSession(student, seed=seed, meld=meld, model_seat=model_seat,
                    human_seat=human_seat, spectate=spectate,
                    student_margin=_state["args"].student_margin)
    _state["game"] = g
    _state["spectate"] = spectate
    # If the model is on move first (model-first versus), let it play once.
    if not spectate and g.current == g.model_seat and g.outcome is None:
        g.model_turn()
    return g


def _build_response():
    g = _state["game"]
    spectate = _state["spectate"]
    resp = {
        "state": g.public_state(),
        "model_decision": g.last_model,
        "human_options": None,
    }
    if (not spectate) and g.outcome is None and g.current == g.human_seat:
        resp["human_options"] = g.human_candidates()
    return resp


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML_PATH, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with _lock:
                if _state["game"] is None:
                    self._send(200, {"state": None})
                else:
                    self._send(200, _build_response())
        else:
            self._send(404, {"error": "not found"})

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        try:
            payload = self._read_json()
            with _lock:
                if self.path == "/api/new":
                    _new_game(
                        mode=payload.get("mode", "versus"),
                        seed=int(payload.get("seed", 2000)),
                        meld=int(payload.get("meld", 30)),
                        first=payload.get("first", "human"),
                    )
                    self._send(200, _build_response())
                elif self.path == "/api/human":
                    g = _state["game"]
                    action = int(payload.get("action", MAX_CANDIDATES))
                    if g.outcome is None and g.current == g.human_seat:
                        g.human_move(action)
                        if g.outcome is None and g.current == g.model_seat:
                            g.model_turn()
                    self._send(200, _build_response())
                elif self.path == "/api/step":
                    g = _state["game"]
                    if g.outcome is None:
                        if g.current == g.model_seat:
                            g.model_turn()
                        else:
                            g.greedy_turn()
                    self._send(200, _build_response())
                else:
                    self._send(404, {"error": "not found"})
        except Exception as e:  # surface errors to the browser
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distill_s1s_dagger1.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--student-margin", type=float, default=0.0)
    args = ap.parse_args()

    print(f"loading model {args.model} ...")
    _state["student"] = Student(args.model)
    _state["args"] = args
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Rummikub vs student — open  http://localhost:{args.port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
