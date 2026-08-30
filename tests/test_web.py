from web.app import STATE, app


def test_health_and_state_endpoints():
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"

    with app.app_context():
        STATE["buffer"][:] = ["H", "I"]
    state = client.get("/state")
    assert state.status_code == 200
    assert state.get_json()["text"] == "HI"

    cleared = client.post("/clear")
    assert cleared.status_code == 200
    assert client.get("/state").get_json()["text"] == ""
