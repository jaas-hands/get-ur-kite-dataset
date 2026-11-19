import io
import tarfile

import pytest

from kite_data.sample_audit import import_archive

pytestmark = pytest.mark.no_assets


@pytest.mark.parametrize("name,symlink", [("../escape", False), ("/tmp/escape", False),
                                         ("leap_sample_v1/link", True)])
def test_archive_rejects_paths_and_symlinks_before_writing(tmp_path, name, symlink):
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp"
        tar.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="unsupported archive entry"):
        import_archive(archive, tmp_path / "input")
    assert not (tmp_path / "input").exists()
