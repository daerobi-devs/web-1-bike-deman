"""
Daeroom Cloud Python Helper (Zero External Dependency Beyond requests)
Menghubungkan aplikasi Streamlit/Python ke PostgreSQL, S3 Storage, dan Auth Daeroom Cloud.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


def _load_local_env() -> None:
    """Memuat variabel dari .env.local atau .env jika belum ada di os.environ."""
    base_dir = Path(__file__).resolve().parent
    for env_name in (".env.local", ".env"):
        env_path = base_dir / env_name
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_local_env()


def _get_config() -> Dict[str, str]:
    url = os.environ.get("DAEROOM_URL", "https://apiv1.daeroom.my.id").rstrip("/")
    api_key = os.environ.get("DAEROOM_API_KEY", "")
    readonly_key = os.environ.get("DAEROOM_READONLY_KEY", "")
    s3_url = os.environ.get("NEXT_PUBLIC_S3_URL", "https://s3.daeroom.my.id").rstrip("/")
    s3_bucket = os.environ.get("DAEROOM_S3_BUCKET", "bikesharingdemand-assets")
    return {
        "url": url,
        "api_key": api_key,
        "readonly_key": readonly_key,
        "s3_url": s3_url,
        "s3_bucket": s3_bucket,
    }


def query_db(
    sql: str,
    params: Optional[List[Any]] = None,
    readonly: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Menjalankan parameterized SQL query ke Daeroom Cloud PostgreSQL (/v1/query).
    Gunakan readonly=True untuk query SELECT publik agar memakai DAEROOM_READONLY_KEY.
    """
    cfg = _get_config()
    token = cfg["readonly_key"] if (readonly and cfg["readonly_key"]) else cfg["api_key"]
    if not token:
        raise RuntimeError("Kredensial DAEROOM_API_KEY / DAEROOM_READONLY_KEY tidak ditemukan di environment.")

    resp = requests.post(
        f"{cfg['url']}/v1/query",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"sql": sql, "params": params or []},
        timeout=timeout,
    )
    data = resp.json()
    if resp.status_code >= 400 or not data.get("success", False):
        raise RuntimeError(f"Daeroom Query Error ({resp.status_code}): {data}")
    return data


def upload_asset(
    file_bytes: bytes,
    filename: str,
    custom_path: Optional[str] = None,
    content_type: str = "application/octet-stream",
    timeout: int = 30,
) -> Dict[str, Any]:
    """Mengunggah file ke MinIO S3 Bucket via Daeroom Storage Gateway."""
    cfg = _get_config()
    if not cfg["api_key"]:
        raise RuntimeError("DAEROOM_API_KEY tidak ditemukan di environment.")

    files = {"file": (filename, file_bytes, content_type)}
    form_data = {"path": custom_path} if custom_path else {}
    resp = requests.post(
        f"{cfg['url']}/v1/storage/upload",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        files=files,
        data=form_data,
        timeout=timeout,
    )
    data = resp.json()
    if resp.status_code >= 400:
        raise RuntimeError(f"Daeroom Storage Error ({resp.status_code}): {data}")
    return data


def get_public_url(name: str) -> str:
    """Mengembalikan URL publik aset di bucket S3 Daeroom."""
    cfg = _get_config()
    clean_name = name.lstrip("/")
    return f"{cfg['s3_url']}/{cfg['s3_bucket']}/{clean_name}"


def register_user(email: str, password: str, full_name: Optional[str] = None) -> Dict[str, Any]:
    """Registrasi user baru melalui Daeroom Managed Auth."""
    cfg = _get_config()
    payload: Dict[str, Any] = {"email": email, "password": password}
    if full_name:
        payload["full_name"] = full_name
    resp = requests.post(
        f"{cfg['url']}/v1/auth/register",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=15,
    )
    return resp.json()


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Login user melalui Daeroom Managed Auth."""
    cfg = _get_config()
    resp = requests.post(
        f"{cfg['url']}/v1/auth/login",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        json={"email": email, "password": password},
        timeout=15,
    )
    return resp.json()


def get_user(user_token: str) -> Dict[str, Any]:
    """Ambil profil user aktif berdasarkan x-user-token."""
    cfg = _get_config()
    resp = requests.get(
        f"{cfg['url']}/v1/auth/me",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "x-user-token": user_token,
        },
        timeout=15,
    )
    return resp.json()
