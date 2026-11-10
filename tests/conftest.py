import pytest

from kite_data import config


@pytest.fixture(autouse=True)
def _assets(request):
    if request.node.get_closest_marker("no_assets"):
        return
    if not (config.BASE_URDF.exists() and config.MENAGERIE_XML.exists()):
        pytest.skip("third-party assets missing: run scripts/fetch_assets.sh")
