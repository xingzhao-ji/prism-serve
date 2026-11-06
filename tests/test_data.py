import io
import tarfile

import pytest

from prismserve.data import safe_extract


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", tarfile.REGTYPE), ("x", tarfile.SYMTYPE), ("/escape", tarfile.REGTYPE)],
)
def test_archive_extraction_rejects_traversal_and_links(tmp_path, name, kind):
    source = tmp_path / "bad.tgz"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.type = kind
        info.size = 1 if kind == tarfile.REGTYPE else 0
        info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
        archive.addfile(info, io.BytesIO(b"x") if info.size else None)
    with pytest.raises(ValueError):
        safe_extract(source, tmp_path / "out")
