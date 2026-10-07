from dotenv import load_dotenv

load_dotenv(dotenv_path=".env_sample", verbose=True)


def test_imports() -> None:
    # from celery_task import analyze_stable_series  # TODO
    from body_organ_analysis import (  # noqa
        analyze_ct,
        store_dicoms,
        store_excel,
    )
    from body_organ_analysis.compute.constants import BASE_MODELS  # noqa
    from body_organ_analysis.compute.util import (  # noqa
        ADDITIONAL_MODELS_OUTPUT_NAME,
    )
