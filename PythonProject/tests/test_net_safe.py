"""Тесты egress-политики: приложение не должно ходить во внутреннюю сеть.

Проверяется именно защита от SSRF, а не скачивание реальных картинок:
тесты полностью офлайновые и не требуют сети.
"""

from __future__ import annotations

import unittest

from net_safe import (
    NotImageError,
    UnsafeURLError,
    is_public_address,
    sniff_image_type,
    validate_public_url,
)


class NetSafetyTests(unittest.TestCase):
    def test_allowed_schemes(self) -> None:
        for url in ("https://8.8.8.8/photo.jpg", "http://8.8.8.8/a.png"):
            parsed = validate_public_url(url)
            self.assertIn(parsed.scheme, {"http", "https"})

    def test_dangerous_schemes_are_rejected(self) -> None:
        for url in (
            "file:///C:/Windows/win.ini",
            "ftp://8.8.8.8/photo.jpg",
            "gopher://8.8.8.8:70/_test",
            "data:image/jpeg;base64,AAAA",
            "javascript:alert(1)",
            "http:///nohost",
        ):
            with self.assertRaises(UnsafeURLError, msg=url):
                validate_public_url(url)

    def test_credentials_in_url_are_rejected(self) -> None:
        with self.assertRaises(UnsafeURLError):
            validate_public_url("http://user:pass@8.8.8.8/photo.jpg")

    def test_internal_targets_are_rejected(self) -> None:
        for url in (
            "http://127.0.0.1:8080/admin",
            "http://localhost/secret",
            "http://10.1.2.3/photo.jpg",
            "http://192.168.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://172.16.5.5/x.jpg",
            "http://[::1]/x.jpg",
            "http://0.0.0.0/",
        ):
            with self.assertRaises(UnsafeURLError, msg=url):
                validate_public_url(url)

    def test_private_escape_hatch_is_explicit(self) -> None:
        parsed = validate_public_url("http://127.0.0.1:8080/x.jpg", allow_private=True)
        self.assertEqual(parsed.hostname, "127.0.0.1")

    def test_address_classification(self) -> None:
        self.assertTrue(is_public_address("8.8.8.8"))
        self.assertTrue(is_public_address("93.158.134.3"))
        for address in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1", "224.0.0.1"):
            self.assertFalse(is_public_address(address), address)

    def test_image_sniffing(self) -> None:
        self.assertEqual(sniff_image_type(b"\xff\xd8\xff\xe0" + b"\x00" * 20), "image/jpeg")
        self.assertEqual(sniff_image_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20), "image/png")
        self.assertEqual(sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertIsNone(sniff_image_type(b"<html><body>not an image</body></html>"))

    def test_html_instead_of_image_is_rejected(self) -> None:
        # Низкоуровневая функция проверки магии: HTML не должен пройти как картинка.
        self.assertIsNone(sniff_image_type(b"<!DOCTYPE html><title>Login</title>"))
        self.assertTrue(issubclass(NotImageError, ValueError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
