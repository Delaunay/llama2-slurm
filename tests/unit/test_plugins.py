import llama2.plugins
from llama2.core import discover_plugins


def test_plugins():
    plugins = discover_plugins(llama2.plugins)

    assert len(plugins) == 1
