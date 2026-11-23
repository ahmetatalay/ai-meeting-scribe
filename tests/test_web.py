"""Web UI tests.

The security tests matter most here: this server holds verbatim meeting
transcripts on a port that any page in your browser can reach.
"""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scribe_audio.web.server import (  # noqa: E402
    HOST,
    AppState,
    _hostname_of,
    create_app,
)


@pytest.fixture
def state(tmp_path):
    return AppState(raw_dir=tmp_path)


@pytest.fixture
def client(state):
    # base_url fixes the Host header to a loopback name for the happy path.
    return TestClient(create_app(state), base_url="http://127.0.0.1:8765")


class TestBindAddress:
    def test_host_is_loopback_and_not_configurable(self):
        """No host option exists, so the server cannot be exposed by a flag."""
        assert HOST == "127.0.0.1"

        import inspect

        from scribe_audio.web import server

        assert "0.0.0.0" not in inspect.getsource(server)
        assert "host" not in inspect.signature(server.serve).parameters


class TestHostHeaderValidation:
    """DNS rebinding: a hostname an attacker controls pointed at 127.0.0.1
    is same-origin to the browser, so the Host header has to be checked."""

    @pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1:8765",
                                      "localhost", "127.0.0.1"])
    def test_loopback_hosts_are_accepted(self, state, host):
        c = TestClient(create_app(state))
        res = c.get(f"/api/devices?token={state.token}", headers={"Host": host})
        assert res.status_code == 200

    @pytest.mark.parametrize("host", ["evil.example.com", "attacker.test:8765",
                                      "scribe.attacker.io"])
    def test_foreign_hosts_are_refused(self, state, host):
        c = TestClient(create_app(state))
        res = c.get(f"/api/devices?token={state.token}", headers={"Host": host})
        assert res.status_code == 421

    def test_the_page_itself_is_also_guarded(self, state):
        c = TestClient(create_app(state))
        assert c.get("/", headers={"Host": "evil.example.com"}).status_code == 421

    def test_hostname_parsing(self):
        assert _hostname_of("localhost:8765") == "localhost"
        assert _hostname_of("127.0.0.1") == "127.0.0.1"
        assert _hostname_of("[::1]:8765") == "[::1]"
        assert _hostname_of("") == ""


class TestCsrf:
    """Same-origin policy stops another site reading responses, not sending
    a POST. Without a token, any page you visited could start a recording."""

    def test_post_without_a_token_is_refused(self, client):
        assert client.post("/api/start", json={}).status_code == 403

    def test_post_with_a_wrong_token_is_refused(self, client):
        res = client.post("/api/start", json={},
                          headers={"X-Scribe-Token": "guessed"})
        assert res.status_code == 403

    def test_stop_is_also_protected(self, client):
        assert client.post("/api/stop").status_code == 403

    def test_retitle_is_also_protected(self, client):
        assert client.post("/api/title", json={"title": "x"}).status_code == 403

    def test_get_routes_require_the_token_too(self, client):
        for path in ("/api/devices", "/api/preflight", "/api/status"):
            assert client.get(path).status_code == 403, path
            assert client.get(f"{path}?token=wrong").status_code == 403, path

    def test_event_stream_requires_the_token(self, client):
        assert client.get("/api/events").status_code == 403

    def test_valid_token_is_accepted(self, client, state):
        res = client.get(f"/api/devices?token={state.token}")
        assert res.status_code == 200


class TestTokenDelivery:
    def test_token_is_embedded_in_the_page(self, client, state):
        body = client.get("/").text
        assert state.token in body
        assert "__SCRIBE_TOKEN__" not in body

    def test_token_is_not_served_from_an_api(self, client, state):
        """A rebinding attack that could fetch the token would defeat it."""
        for path in ("/api/devices", "/api/preflight", "/api/status"):
            assert state.token not in client.get(f"{path}?token={state.token}").text

    def test_tokens_differ_between_runs(self, tmp_path):
        a = AppState(raw_dir=tmp_path).token
        b = AppState(raw_dir=tmp_path).token
        assert a != b and len(a) > 30


class TestHardeningHeaders:
    def test_page_is_not_cached_and_has_a_csp(self, client):
        res = client.get("/")
        assert res.headers["Cache-Control"] == "no-store"
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'none'" in res.headers["Content-Security-Policy"]

    def test_no_cors_headers_are_granted(self, client, state):
        res = client.get(f"/api/devices?token={state.token}",
                         headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in {
            k.lower() for k in res.headers
        }

    def test_api_docs_are_disabled(self, client):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404, path


class TestFrontEnd:
    """Guards on the served page.

    The device pickers once both defaulted to whichever input happened to be
    first, because `new Option(text, value, defaultSelected, selected)` sets
    selectedness on a *detached* option and browsers do not honour that
    reliably. Selection has to be assigned after the options are in the DOM.
    """

    @pytest.fixture
    def page(self, client):
        return client.get("/").text

    def test_selection_is_assigned_after_populating(self, page):
        assert "sys.value = String(devices.loopback_id)" in page
        assert "mic.value = String(devices.default_input)" in page

    def test_detached_option_selection_is_not_used(self, page):
        # A 4-argument new Option(...) call is the unreliable form.
        import re

        for call in re.findall(r"new Option\(([^)]*)\)", page):
            assert call.count(",") <= 1, f"4-arg new Option: {call}"

    def test_same_device_for_both_streams_is_guarded(self, page):
        assert "validateSelection" in page
        assert "pair-warn" in page

    def test_stale_meter_warning_is_cleared_when_not_recording(self, page):
        assert 'dead-sys").hidden = true' in page

    def test_hidden_attribute_is_not_defeated_by_author_styles(self, page):
        """`hidden` comes from the UA stylesheet, and author CSS wins over the
        UA origin regardless of specificity. Without an explicit reset, any
        `display` rule un-hides its element — which left the "no meeting audio"
        warning and the empty footer visible permanently."""
        import re

        css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
        reset = re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css)
        assert reset, "missing [hidden] { display: none !important } reset"

        # Elements that rely on the attribute must be declared with it.
        for element_id in ("dead-sys", "meta", "pair-warn"):
            match = re.search(rf'id="{element_id}"[^>]*>', page)
            assert match and "hidden" in match.group(0), element_id

    def test_the_page_itself_never_scrolls(self, page):
        """New speech must not move the whole document.

        The transcript is its own scroll region; scrolling the window instead
        made the page jump away from whatever was being read.
        """
        assert "window.scrollTo" not in page

        import re

        css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
        turns = re.search(r"#turns\s*\{([^}]*)\}", css)
        assert turns and "overflow-y: auto" in turns.group(1)

    def test_auto_scroll_only_follows_from_the_bottom(self, page):
        """Someone who scrolled up to read must not be yanked down."""
        assert "atBottom" in page
        assert "STICK_THRESHOLD_PX" in page
        # The decision has to be taken before appending, since appending
        # changes the geometry it depends on.
        before_append = page.index("const following = atBottom")
        append = page.index("what.append(span)")
        assert before_append < append

    def test_a_jump_to_latest_control_exists(self, page):
        import re

        tag = re.search(r'id="jump"[^>]*>', page)
        assert tag and "hidden" in tag.group(0)
        assert "scrollToLatest" in page

    def test_every_element_the_script_looks_up_exists(self, page):
        """A typo in an id is silent at load time and only fails when that code
        path runs, so check the whole set up front."""
        import re

        wanted = set(re.findall(r'\$\("([^"]+)"\)', page))
        declared = set(re.findall(r'id="([^"]+)"', page))
        missing = wanted - declared
        assert not missing, f"script references non-existent ids: {sorted(missing)}"

    def test_library_view_is_present(self, page):
        for element_id in ("tab-record", "tab-library", "search", "items",
                           "reader", "reader-title", "reader-body"):
            assert f'id="{element_id}"' in page, element_id

    def test_deleting_from_the_ui_asks_for_confirmation(self, page):
        """Deletion is irreversible — there is no trash."""
        assert "confirm(" in page
        delete_fn = page[page.index("async function deleteTranscript"):]
        confirm_at = delete_fn.index("confirm(")
        request_at = delete_fn.index('method: "DELETE"')
        assert confirm_at < request_at, "the request must not precede the confirm"

    def test_user_text_is_never_interpolated_into_html(self, page):
        """Titles and transcript text come from files; they go in as text."""
        import re

        # Inside the library/reader renderers, content is assigned via
        # textContent rather than innerHTML.
        section = page[page.index("async function loadLibrary"):
                       page.index("/* ------------------------------------------------------------------ actions */")]
        for assignment in re.findall(r"\.innerHTML\s*=\s*([^;]+);", section):
            assert "t." not in assignment and "data." not in assignment, assignment

    def test_page_is_self_contained(self, page):
        """The CSP forbids external loads, so the page must not request any."""
        import re

        external = re.findall(r'(?:src|href)\s*=\s*["\'](https?:)?//', page)
        assert external == []

    def test_no_template_placeholder_remains(self, page):
        assert "__SCRIBE_TOKEN__" not in page


class TestDevicesAndPreflight:
    def test_devices_lists_inputs_and_outputs(self, client, state):
        data = client.get(f"/api/devices?token={state.token}").json()
        assert "inputs" in data and "outputs" in data
        assert "loopback_id" in data and "default_input" in data

    def test_preflight_returns_checks(self, client, state):
        data = client.get(f"/api/preflight?token={state.token}").json()
        ids = {c["id"] for c in data["checks"]}
        assert ids == {"loopback", "routing", "microphone"}
        assert "Same as System" in data["note"]


class TestRecordingRoutes:
    def test_status_is_idle_before_anything_starts(self, client, state):
        data = client.get(f"/api/status?token={state.token}").json()
        assert data["state"] == "idle"

    def test_stop_with_nothing_running_is_a_conflict(self, client, state):
        res = client.post("/api/stop", headers={"X-Scribe-Token": state.token})
        assert res.status_code == 409

    def test_start_rejects_a_request_with_no_devices(self, client, state):
        res = client.post(
            "/api/start",
            json={"mic_device": None, "system_device": None},
            headers={"X-Scribe-Token": state.token},
        )
        assert res.status_code == 400

    def test_start_rejects_a_second_recording(self, client, state, monkeypatch):
        from scribe_audio.web import server as mod

        class Fake:
            def __init__(self, *a, **k):
                self.state = "recording"
                self.is_running = True
                self.writer = None
                self.capture = None

            def start(self):
                pass

            def stats_dict(self):
                return {}

        monkeypatch.setattr(mod, "RecordingSession", Fake)
        headers = {"X-Scribe-Token": state.token}
        first = client.post("/api/start", json={"mic_device": 0}, headers=headers)
        assert first.status_code == 200
        second = client.post("/api/start", json={"mic_device": 0}, headers=headers)
        assert second.status_code == 409

    def test_retitle_without_a_session_is_a_conflict(self, client, state):
        res = client.post("/api/title", json={"title": "Review"},
                          headers={"X-Scribe-Token": state.token})
        assert res.status_code == 409

    def test_retitle_requires_a_non_empty_title(self, client, state):
        res = client.post("/api/title", json={"title": "  "},
                          headers={"X-Scribe-Token": state.token})
        assert res.status_code == 400


class TestLibraryRoutes:
    @pytest.fixture
    def seeded(self, tmp_path):
        """A library directory with two transcripts, plus a file outside it."""
        import json as _json

        raw = tmp_path / "raw"
        raw.mkdir()
        for name, title in (("transcript_2026-11-23_1000.md", "Sprint planning"),
                            ("transcript_2026-11-22_0900.md", "One to one")):
            (raw / name).write_text(
                f"---\ntitle: {title}\nstarted: 2026-11-2{name[14]}T10:00:00\n---\n\n"
                f"# {title}\n\n**You**: something about migrations\n",
                encoding="utf-8",
            )
            (raw / name).with_suffix(".json").write_text(
                _json.dumps({"title": title, "markdown": name, "segments": []}),
                encoding="utf-8",
            )
        (tmp_path / "outside.md").write_text("private", encoding="utf-8")
        return raw

    @pytest.fixture
    def lib_client(self, seeded):
        st = AppState(raw_dir=seeded)
        return TestClient(create_app(st), base_url="http://127.0.0.1:8765"), st

    def test_listing(self, lib_client):
        client, st = lib_client
        data = client.get(f"/api/transcripts?token={st.token}").json()
        assert len(data["transcripts"]) == 2
        assert {t["title"] for t in data["transcripts"]} == {
            "Sprint planning", "One to one",
        }

    def test_listing_requires_the_token(self, lib_client):
        client, _ = lib_client
        assert client.get("/api/transcripts").status_code == 403

    def test_search(self, lib_client):
        client, st = lib_client
        data = client.get(f"/api/transcripts?token={st.token}&q=sprint").json()
        assert len(data["transcripts"]) == 1

    def test_reading_one(self, lib_client):
        client, st = lib_client
        data = client.get(
            f"/api/transcripts/transcript_2026-11-23_1000.md?token={st.token}"
        ).json()
        assert data["title"] == "Sprint planning"
        assert data["turns"][0]["speaker"] == "You"

    def test_reading_a_missing_one(self, lib_client):
        client, st = lib_client
        res = client.get(f"/api/transcripts/transcript_1999-01-01_0000.md?token={st.token}")
        assert res.status_code == 404

    @pytest.mark.parametrize("name", ["..%2F..%2Foutside.md", "...md", "secrets.json"])
    def test_traversal_and_bad_names_are_refused(self, lib_client, name):
        client, st = lib_client
        res = client.get(f"/api/transcripts/{name}?token={st.token}")
        assert res.status_code in (400, 404), res.status_code

    def test_rename(self, lib_client):
        client, st = lib_client
        res = client.post(
            "/api/transcripts/transcript_2026-11-23_1000.md/title",
            json={"title": "Quarterly review"},
            headers={"X-Scribe-Token": st.token},
        )
        assert res.status_code == 200
        listing = client.get(f"/api/transcripts?token={st.token}").json()
        assert "Quarterly review" in {t["title"] for t in listing["transcripts"]}

    def test_rename_requires_the_token(self, lib_client):
        client, _ = lib_client
        res = client.post("/api/transcripts/transcript_2026-11-23_1000.md/title",
                          json={"title": "x"})
        assert res.status_code == 403

    def test_rename_rejects_an_empty_title(self, lib_client):
        client, st = lib_client
        res = client.post("/api/transcripts/transcript_2026-11-23_1000.md/title",
                          json={"title": " "}, headers={"X-Scribe-Token": st.token})
        assert res.status_code == 400

    def test_delete(self, lib_client, seeded):
        client, st = lib_client
        res = client.delete("/api/transcripts/transcript_2026-11-23_1000.md",
                            headers={"X-Scribe-Token": st.token})
        assert res.status_code == 200
        assert not (seeded / "transcript_2026-11-23_1000.md").exists()
        assert (seeded / "transcript_2026-11-22_0900.md").exists()

    def test_delete_requires_the_token(self, lib_client, seeded):
        """Without this, any page you had open could wipe your transcripts."""
        client, _ = lib_client
        res = client.delete("/api/transcripts/transcript_2026-11-23_1000.md")
        assert res.status_code == 403
        assert (seeded / "transcript_2026-11-23_1000.md").exists()

    def test_delete_cannot_escape_the_library(self, lib_client, seeded):
        client, st = lib_client
        outside = seeded.parent / "outside.md"
        res = client.delete("/api/transcripts/..%2Foutside.md",
                            headers={"X-Scribe-Token": st.token})
        assert res.status_code in (400, 404)
        assert outside.exists()

    def test_delete_refuses_the_recording_in_progress(self, lib_client, seeded,
                                                      monkeypatch):
        client, st = lib_client
        name = "transcript_2026-11-23_1000.md"

        class FakeWriter:
            path = seeded / name
            title = "live"

        class FakeSession:
            is_running = True
            writer = FakeWriter()
            state = "recording"
            capture = None

            def stats_dict(self):
                return {}

        st.session = FakeSession()
        res = client.delete(f"/api/transcripts/{name}",
                            headers={"X-Scribe-Token": st.token})
        assert res.status_code == 409
        assert (seeded / name).exists()


class TestEventStream:
    def test_stream_opens_and_sends_sse(self, client, state):
        with client.stream("GET", f"/api/events?token={state.token}") as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            for line in res.iter_lines():
                if line.startswith("data:"):
                    assert json.loads(line[5:])["type"] == "hello"
                    break


class TestPayloadCoercion:
    def test_device_and_language_coercion(self):
        from scribe_audio.web.server import _as_device, _as_language

        assert _as_device(None) is None
        assert _as_device("") is None
        assert _as_device("3") == 3
        assert _as_device(3) == 3

        assert _as_language("auto") is None
        assert _as_language("") is None
        assert _as_language("tr") == "tr"
