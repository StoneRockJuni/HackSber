"""Fetching a remote image *safely*.

The prototype lets a user paste any link ("проверь вот это фото из интернета").
That single feature is also the most dangerous one in a bank network: a naive
``urllib.request.urlopen`` turns the app into a proxy for the internal network
(SSRF - Server-Side Request Forgery). An attacker submits
``http://10.0.0.5/admin`` or ``http://169.254.169.254/latest/meta-data/`` and
reads the response through the report, or uses the box to scan ports.

What this module enforces
-------------------------
1. Only ``http`` and ``https``. No ``file:``, ``ftp:``, ``data:``, ``gopher:``.
2. No credentials in the URL (``http://user:pass@host``).
3. The host name is resolved and **every** resulting address must be public -
   private, loopback, link-local, multicast and reserved ranges are refused.
   ``localhost`` and ``*.internal`` die here too.
4. Every redirect hop is re-validated, so a public URL cannot bounce to
   ``127.0.0.1``.
5. Hard limits on size, time and number of redirects.
6. The payload must actually be an image (magic bytes), not an HTML page with
   a misleading ``Content-Type``.

Residual risk: DNS rebinding (the name resolves to a public IP during the
check and to a private one at connect time) is not fully solved here. For a
production pilot, resolve once, pin the IP and use it for the connection, or
put the fetcher behind an egress proxy with an allow-list.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT = 20.0
MAX_REDIRECTS = 3
USER_AGENT = "StroyKonturInspector/2.0 (+local, on-prem)"

ALLOWED_SCHEMES = {"http", "https"}
_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


class UnsafeURLError(ValueError):
    """The URL is not allowed by the egress policy."""


class NotImageError(ValueError):
    """The downloaded bytes are not an image."""


@dataclass
class FetchedImage:
    data: bytes
    content_type: str
    final_url: str
    declared_type: str | None = None


def _allow_private() -> bool:
    """Escape hatch for local tests only (never enable in a bank contour)."""
    return os.getenv("CHECK_PHOTO_ALLOW_PRIVATE", "0").strip() == "1"


def is_public_address(address: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(url: str, allow_private: bool | None = None) -> urllib.parse.ParseResult:
    """Validate scheme, host and resolved addresses. Raises UnsafeURLError."""
    if allow_private is None:
        allow_private = _allow_private()
    try:
        parsed = urllib.parse.urlsplit((url or "").strip())
    except ValueError as exc:
        raise UnsafeURLError(f"Некорректный URL: {exc}") from exc
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(
            f"Разрешены только http/https, получено: {parsed.scheme or 'пусто'}"
        )
    if not parsed.hostname:
        raise UnsafeURLError("В URL нет имени хоста")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URL с логином/паролем запрещён")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError(f"Некорректный порт: {exc}") from exc
    if allow_private:
        return parsed
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Не удалось разрешить имя {parsed.hostname}: {exc}") from exc
    if not infos:
        raise UnsafeURLError(f"Имя {parsed.hostname} не разрешилось ни в один адрес")
    for info in infos:
        address = info[4][0]
        if not is_public_address(address):
            raise UnsafeURLError(
                f"Адрес {address} ({parsed.hostname}) не является публичным: "
                "внутренние, локальные и служебные сети запрещены (защита от SSRF)"
            )
    return parsed


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate the target of every redirect instead of trusting it."""

    max_redirections = MAX_REDIRECTS

    def __init__(self, allow_private: bool) -> None:
        self._allow_private = allow_private
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        validate_public_url(newurl, allow_private=self._allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sniff_image_type(data: bytes) -> str | None:
    """Detect the real image type from magic bytes (Content-Type can lie)."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    return None


def fetch_image(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    allow_private: bool | None = None,
) -> FetchedImage:
    """Download an image with the egress policy above applied.

    Raises UnsafeURLError / NotImageError / urllib.error.URLError.
    """
    if allow_private is None:
        allow_private = _allow_private()
    validate_public_url(url, allow_private=allow_private)

    opener = urllib.request.build_opener(_SafeRedirectHandler(allow_private))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
    )
    with opener.open(request, timeout=timeout) as response:
        declared = response.headers.get_content_type()
        final_url = response.geturl()
        # Redirects inside opener.open are validated, but the final URL is
        # validated once more for defence in depth.
        validate_public_url(final_url, allow_private=allow_private)
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise NotImageError(f"Файл больше лимита {max_bytes // (1024 * 1024)} МБ")
    if not data:
        raise NotImageError("Сервер вернул пустой ответ")
    detected = sniff_image_type(data)
    if detected is None:
        raise NotImageError(
            f"Ответ не является изображением (Content-Type: {declared}). "
            "Скачивайте прямую ссылку на JPG/PNG/WebP, а не страницу с фото."
        )
    return FetchedImage(data=data, content_type=detected, final_url=final_url, declared_type=declared)


def fetch_ok(url: str) -> tuple[bool, str]:
    """Non-raising wrapper used by the UI: (ok, human message)."""
    try:
        fetched = fetch_image(url)
    except (UnsafeURLError, NotImageError) as exc:
        return False, str(exc)
    except urllib.error.HTTPError as exc:
        return False, f"Сервер ответил HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"Сеть недоступна: {exc.reason}"
    except socket.timeout:
        return False, "Таймаут при скачивании"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Ошибка загрузки: {exc}"
    return True, f"OK ({len(fetched.data) / 1024:.0f} КБ, {fetched.content_type})"


__all__ = [
    "FetchedImage",
    "NotImageError",
    "UnsafeURLError",
    "fetch_image",
    "fetch_ok",
    "is_public_address",
    "sniff_image_type",
    "validate_public_url",
]
