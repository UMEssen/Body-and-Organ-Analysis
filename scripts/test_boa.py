import logging
import unittest

from dotenv import load_dotenv

logger = logging.getLogger()
logger.setLevel(logging.INFO)
load_dotenv(dotenv_path=".env_sample", verbose=True)


class BasicTests(unittest.TestCase):
    @staticmethod
    def test_imports() -> None:
        from celery_task import analyze_stable_series  # noqa

        from body_organ_analysis import analyze_ct  # noqa
        from body_organ_analysis import store_dicoms, store_excel  # noqa
        from body_organ_analysis.compute.constants import BASE_MODELS  # noqa
        from body_organ_analysis.compute.util import (  # noqa
            ADDITIONAL_MODELS_OUTPUT_NAME,
        )


if __name__ == "__main__":
    unittest.main(exit=False)
