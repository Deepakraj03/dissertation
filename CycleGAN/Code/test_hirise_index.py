import csv
import io

import pytest

from hirise_index import parse_index_row

# Real sample row from RDRCUMINDEX.TAB (AEB_000001_0150), verified against
# https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB
SAMPLE_ROW = (
    '"MROHR_0001","RDR/AEB/ORB_000000_000099/AEB_000001_0150/'
    'AEB_000001_0150_RED.JP2  ","MRO","HIRISE","AEB_000001_0150",'
    '"AEB_000001_0150_RED  ","2  ","MARS                            ",'
    '     1,"Aerobraking                   ",'
    '"Sample of Argyre Basin rim                                                 ",'
    '"2006-03-24T04:50:31     ","827643049:47201 ",'
    '"2006-03-24T04:50:31     ","827643050:37396 ",'
    '"2006-03-24T04:51:02     ","827643081:07204 ", 53911, 29279, '
    '0.35607,87.0475, 87.2081,1469.320,4853.87,1469.670,  270.0000,  '
    '336.9280,   12.0600,    9.5812,  -52.3105,  300.7610,   1.62365,'
    '    29.396,    7.4125,"NO ",  -52.8767,  -51.5351,  300.2040,  '
    '301.3370, 1.47, 40183.102,"EQUIRECTANGULAR    ",-50.0, 180.000,  '
    '-2070840.0,  -3104770.0,  -51.5351,  300.5280,  -51.6056,  '
    '301.3370,  -52.8767,  301.0040,  -52.8092,  300.2040'
)


def test_parse_index_row_extracts_correct_footprint():
    row = next(csv.reader(io.StringIO(SAMPLE_ROW), skipinitialspace=True))
    fp = parse_index_row(row)

    assert fp.obs_id == "AEB_000001_0150"
    assert abs(fp.min_lat - (-52.8767)) < 1e-4
    assert abs(fp.max_lat - (-51.5351)) < 1e-4
    # Longitudes stored 0-360 in the index; parse_index_row must normalise
    # to -180..180 (300.2040 -> -59.7960, 301.3370 -> -58.6630).
    assert abs(fp.min_lon - (-59.7960)) < 1e-4
    assert abs(fp.max_lon - (-58.6630)) < 1e-4
    assert fp.projection == "EQUIRECTANGULAR"
    # Needed to construct browse-image URLs directly from the index,
    # replacing the broken filename-based lat-code crawl in
    # download_hirise.py (see project history: that approach pulled
    # south-polar images regardless of requested region).
    assert fp.file_name_spec == "RDR/AEB/ORB_000000_000099/AEB_000001_0150/AEB_000001_0150_RED.JP2"


def test_download_index_does_not_leave_partial_file_on_failure(tmp_path, monkeypatch):
    # Reproduces a real failure hit during the live smoke test: the
    # connection reset mid-stream, leaving a 16MB truncated file that
    # exceeded the old ">10MB means already downloaded" check, so a
    # retry would have silently treated it as complete and parsed a
    # corrupted index forever after.
    import hirise_index

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial data" * 100
            raise ConnectionError("simulated network failure")

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(hirise_index.requests, "get", fake_get)

    with pytest.raises(ConnectionError):
        hirise_index.download_index(tmp_path)

    dest_path = tmp_path / "RDRCUMINDEX.TAB"
    assert not dest_path.exists()
