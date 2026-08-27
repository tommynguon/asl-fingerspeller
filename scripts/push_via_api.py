"""Push HEAD to GitHub via GraphQL when git-receive-pack is blocked (corp proxy)."""
from __future__ import annotations

import base64
import json
import ssl
import subprocess
import urllib.error
import urllib.request

OWNER = "tommynguon"
REPO = "asl-fingerspeller"
ZERO = "0" * 40


def token() -> str:
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


def api_graphql(payload: dict) -> dict:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token()}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "asl-fingerspeller",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read()[:800].decode('utf-8', 'ignore')}")


def additions() -> list[dict]:
    files = [p for p in subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0") if p]
    out = []
    for raw in files:
        path = raw.decode("utf-8")
        blob = subprocess.check_output(["git", "show", f"HEAD:{path}"])
        out.append({"path": path, "contents": base64.b64encode(blob).decode()})
    return out


def head_oid() -> str:
    data = api_graphql(
        {
            "query": """
            query($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) {
                defaultBranchRef { target { oid } }
              }
            }
            """,
            "variables": {"owner": OWNER, "name": REPO},
        }
    )
    ref = (data.get("data") or {}).get("repository") or {}
    branch = ref.get("defaultBranchRef")
    if not branch:
        return ZERO
    return branch["target"]["oid"]


def main() -> int:
    oid = head_oid()
    files = additions()
    message = subprocess.check_output(["git", "log", "-1", "--format=%s"], text=True).strip()
    print(f"expectedHeadOid={oid} files={len(files)}")
    result = api_graphql(
        {
            "query": """
            mutation($input: CreateCommitOnBranchInput!) {
              createCommitOnBranch(input: $input) {
                commit { oid url }
              }
            }
            """,
            "variables": {
                "input": {
                    "branch": {
                        "repositoryNameWithOwner": f"{OWNER}/{REPO}",
                        "branchName": "main",
                    },
                    "message": {"headline": message},
                    "fileChanges": {"additions": files},
                    "expectedHeadOid": oid,
                }
            },
        }
    )
    print(json.dumps(result, indent=2)[:4000])
    if result.get("errors"):
        return 1
    print("pushed", result["data"]["createCommitOnBranch"]["commit"]["oid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
