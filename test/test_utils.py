import pytest
from pydantic import BaseModel

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

    code1 = _NSL(network="6E", station="TE234", location="AB")
    code2 = _NSL(network="6E", station="TE234", location="")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE", location="")
    code2 = _NSL(network="6E", station="TE234", location="")
    assert not code1.match(code2)

    code1 = _NSL(network="6E", station="TE", location="")
    code2 = _NSL(network="5E", station="TE234", location="")
    assert not code1.match(code2)
