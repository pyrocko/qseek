import platform
import warnings

import pytest
from pydantic import BaseModel

from qseek import utils
from qseek.utils import _NSL, NSL


def test_nsl():
    nsl_id = "6E.TE234."
    nsl = NSL(*nsl_id.split("."))

    assert nsl.network == "6E"
    assert nsl.station == "TE234"
    assert nsl.location == ""

    class Model(BaseModel):
        nsl: NSL
        nsl_list: list[NSL]

    Model(nsl=nsl, nsl_list=[nsl, nsl, nsl])

    json = """
    {
        "nsl": "6E.TE234.",
        "nsl_list": ["6E.TE234.", "6E.TE234.", "6E.TE234.", ["6E", "TY123", ""]]
    }
    """
    Model.model_validate_json(json)

    json = """
    {
        "nsl": "6E.TE234.",
        "nsl_list": [".TE232"]
    }
    """
    Model.model_validate_json(json)

    json_tpl = """
    {{
        "nsl": "{code}",
        "nsl_list": ["{code}"]
    }}
    """

    invalid_codes = ["6E5.", "6E.TE123112"]

    for code in invalid_codes:
        with pytest.raises(ValueError):
            Model.model_validate_json(json_tpl.format(code=code))

    net_code = _NSL(network="6E", station="", location="")
    sta_code = _NSL(network="6E", station="TE234", location="")
    assert net_code.match(sta_code)

    code1 = _NSL(network="6E", station="TE234", location="")
    code2 = _NSL(network="6E", station="TE234", location="")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="TE234", location="AB")
    code2 = _NSL(network="6E", station="TE234", location="AB")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="TE*", location="AB")
    code2 = _NSL(network="6E", station="TE234", location="AB")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="TE???", location="AB")
    code2 = _NSL(network="6E", station="TE234", location="AB")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="TE234", location="A?")
    code2 = _NSL(network="6E", station="TE234", location="AB")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="TE???", location="AC")
    code2 = _NSL(network="6E", station="TE234", location="AB")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE???")
    code2 = _NSL(network="6E", station="TE234")
    assert code1.match(code2)

    code1 = _NSL(network="6E", station="")
    code2 = _NSL(network="6E", station="")
    assert code1.match(code2)

    code1 = _NSL(network="6?", station="")
    code2 = _NSL(network="6E", station="")
    assert code1.match(code2)

    code1 = _NSL(network="5?", station="")
    code2 = _NSL(network="6E", station="")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE234", location="AB")
    code2 = _NSL(network="6E", station="TE234", location="")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE", location="")
    code2 = _NSL(network="6E", station="TE234", location="")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE", location="")
    code2 = _NSL(network="5E", station="TE234", location="")
    assert not code1.match(code2)


def test_nsl_exclusion():
    exclude_nsls = {_NSL.parse("6E.TE234."), _NSL.parse("6E.TE235.")}
    nsls = [
        _NSL.parse("6E.TE234."),
        _NSL.parse("6E.TE235."),
        _NSL.parse("6E.TE236."),
        _NSL.parse("6E.TE237."),
    ]
    filtered_nsls = [
        nsl for nsl in nsls if not any(ex_nsl.match(nsl) for ex_nsl in exclude_nsls)
    ]

    assert filtered_nsls == [_NSL.parse("6E.TE236."), _NSL.parse("6E.TE237.")]


@pytest.mark.parametrize(
    "build_flags,cpu_features,warns",
    [
        (("-mavx2", "-mfma"), {"sse2", "avx2", "fma"}, False),
        (("-mavx2", "-mfma"), {"sse2", "avx"}, True),
        (("-mavx2", "-mfma"), {"sse2", "avx2"}, True),
        ((), {"sse2"}, False),
        (("-mavx2", "-mfma"), None, False),
    ],
)
def test_check_simd_support(monkeypatch, build_flags, cpu_features, warns):
    monkeypatch.setattr(utils, "get_cpu_features", lambda: cpu_features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        utils.check_simd_support(build_flags)
    assert bool(caught) == warns
    if warns:
        assert caught[0].category is RuntimeWarning
        assert "install qseek from source" in str(caught[0].message)


def test_get_cpu_features():
    features = utils.get_cpu_features()
    if platform.system() in ("Linux", "Darwin") and platform.machine() == "x86_64":
        assert features
        assert "sse2" in features
    if features is not None:
        assert all(feature == feature.lower() for feature in features)
