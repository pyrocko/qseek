from pydantic import BaseModel
from qseek.utils import NSL


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
        "nsl_list": ["6E.TE234.", "6E.TE234.", "6E.TE234."]
    }
    """
    Model.model_validate_json(json)
