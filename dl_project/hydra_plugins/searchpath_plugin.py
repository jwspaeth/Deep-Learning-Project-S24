# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class DLProjectPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        # search_path.append(
        #     provider="dl-project-searchpath-plugin", path="pkg://arbitrary_package/conf"
        # )
        search_path.append(
            provider="dl-project-searchpath-plugin", path="file://config"
        )
        current_dir = os.path.abspath("./")
        search_path.append(
            provider="dl-project-searchpath-plugin", path=f"file://{current_dir}"
        )
