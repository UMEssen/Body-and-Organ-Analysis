import importlib
from pathlib import Path
from typing import Any

import pytest

# skip if the pacs extra is not installed
pytest.importorskip("psycopg2")
util = importlib.import_module("util")
normalize_string = util._normalize_string
allowed_file_names = util._allowed_file_names
resolve_umlauts = util._resolve_umlauts
remove_accents = util._remove_accents
normalize_file_name = util.normalize_file_name

TAGS = {
    "CalledAET": "BOA",
    "StudyDate": "20240101",
    "AccessionNumber": "ACC1",
    "StudyDescription": "CT",
    "SeriesNumber": "3",
    "SeriesDescription": "Thorax",
    "PatientName": "Doe",
    "PatientBirthDate": "19700101",
}
# codespell:ignore-begin
GERMAN = (
    "Äußerst faszinierend illustriert das Möbiusband, wie Geodäten "
    "Örtlichkeiten überbrücken und Übereinstimmungen zwischen einer nicht "
    "orientierbaren Fläche und höherer Topologie enthüllen, während ätherische "
    "Krümmungskurven die überraschend ökonomische, überaus schöne Struktur "
    "verflochten halten, sodaß selbst die kühnsten Köpfe grübelnd das volle "
    "Maß dieser Komplexität zu ermeßen suchen."
)
GERMAN_TRANSLITERATED = (
    "Aeusserst faszinierend illustriert das Moebiusband, wie Geodaeten "
    "Oertlichkeiten ueberbruecken und Uebereinstimmungen zwischen einer nicht "
    "orientierbaren Flaeche und hoeherer Topologie enthuellen, waehrend "
    "aetherische Kruemmungskurven die ueberraschend oekonomische, ueberaus "
    "schoene Struktur verflochten halten, sodass selbst die kuehnsten Koepfe "
    "gruebelnd das volle Mass dieser Komplexitaet zu ermessen suchen."
)
# codespell:ignore-end
FILE_NAME_CASES = [
    ("Müller Öl Übung", "Mueller_Oel_Uebung"),
    ("Straße", "Strasse"),
    ("Tüür", "Tueuer"),
    ("café déjà vu naïve", "cafe_deja_vu_naive"),
    ("folder/sub\\file.txt", "folder_sub_file.txt"),
    ("  my  file   name  ", "my_file_name"),
    ("résumé (final) v2!.JPG", "resume_final_v2.JPG"),
    ("report@2024#final$.pdf", "report2024final.pdf"),
    ("___leading__and__trailing___", "leading_and_trailing"),
    ("Größe: 50% — Tüür\\Straße/Test", "Groesse_50_Tueuer_Strasse_Test"),
    ("a.b,c-d_e", "a.b,c-d_e"),
    ("ÄÖÜäöüß", "AeOeUeaeoeuess"),
    ("山田太郎.png", "unnamed.png"),
    ("Mixed 123 ABC äöü .tar.gz", "Mixed_123_ABC_aeoeue_.tar.gz"),
    (
        "Café Müller & Sons/Straße 12 (draft).PNG",
        "Cafe_Mueller_Sons_Strasse_12_draft.PNG",
    ),
    ("Mu\u0308ller", "Mueller"),
    ("Łukasz Wałęsa", "Lukasz_Walesa"),
    ("Y\u0131ld\u0131z", "Yildiz"),
    ("GROẞE", "GROSSE"),
    ("Đorđe", "Dorde"),
    ("cœur", "coeur"),
    ("Ålborg Ærø", "Aalborg_Aeroe"),
    ("Þór", "Thor"),
    ("my\tfile\nname.txt", "my_file_name.txt"),
    ("my\u00a0file", "my_file"),
    ("", "unnamed"),
    (".", "unnamed"),
    ("..", "unnamed"),
    ("...", "unnamed"),
    ("山田太郎", "unnamed"),
    ("file.", "file"),
    ("file._", "file"),
    ("file.\t", "file"),
    ("a_.", "a"),
    ("report $.", "report"),
]


def test_normalize_string() -> None:
    assert normalize_string("_ _a__b  c_ d _e_f g _ ") == "a_b_c_d_e_f_g"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my file (1)?.txt", "my file 1.txt"),
        ("a\tb\nc", "abc"),
        ("report $.", "report"),
        ("name. . ", "name"),
    ],
)
def test_allowed_file_names(tmp_path: Path, text: str, expected: str) -> None:
    # The name must also survive being created on the file system unchanged
    (tmp_path / allowed_file_names(text)).touch()
    assert [p.name for p in tmp_path.iterdir()] == [expected]


def test_allowed_file_names_ignore_chars() -> None:
    # ignore_chars can let unsafe characters through, so no file is created
    assert allowed_file_names("a\tb", r"\s") == "a\tb"


@pytest.mark.parametrize(
    ("text", "backwards", "expected"),
    [
        (GERMAN, False, GERMAN_TRANSLITERATED),
        (GERMAN_TRANSLITERATED, True, GERMAN),
        # Decomposed umlaut ("u" + combining diaeresis)
        ("Mu\u0308ller", False, "Mueller"),
        # Non-German letters are transliterated forward only
        (
            "Łukasz Y\u0131ld\u0131z GROẞE Ærø Ålborg Þór",
            False,
            "Lukasz Yildiz GROSSE Aeroe Aalborg Thór",
        ),
        (
            "Aachen Thomas Dieter Kolberg Ruediger",
            True,
            "Aachen Thomas Dieter Kolberg Rüdiger",
        ),
    ],
)
def test_resolve_umlauts(text: str, backwards: bool, expected: str) -> None:
    assert resolve_umlauts(text, backwards=backwards) == expected


def test_remove_accents() -> None:
    assert (
        remove_accents("café, déjà vu, über, façade, piñata, rôle")
        == "cafe, deja vu, uber, facade, pinata, role"
    )


@pytest.mark.parametrize(("text", "expected"), FILE_NAME_CASES)
def test_normalize_file_name(text: str, expected: str) -> None:
    assert normalize_file_name(text) == expected


@pytest.mark.parametrize("name", sorted({name for _, name in FILE_NAME_CASES}))
def test_normalize_file_name_is_idempotent(name: str) -> None:
    assert normalize_file_name(name) == name


def test_normalize_file_name_fallback() -> None:
    assert normalize_file_name("", fallback="series") == "series"


@pytest.mark.parametrize(
    ("tags", "infos", "expected"),
    [
        (
            {"StudyDate": "20240101", "AccessionNumber": "ACC/1"},
            ["StudyDate", "AccessionNumber", "StudyDescription"],
            "20240101_ACC_1_UnknownStudyDescription",
        ),
        # DICOM "^" component separator
        (
            {"PatientName": "Doe^John", "PatientBirthDate": "19700101"},
            ["PatientName", "PatientBirthDate"],
            "Doe_John_19700101",
        ),
        # Empty and fully sanitized values
        (
            {"SeriesNumber": "", "SeriesDescription": "???", "StudyDate": None},
            ["SeriesNumber", "SeriesDescription", "StudyDate"],
            "UnknownSeriesNumber_UnknownSeriesDescription_UnknownStudyDate",
        ),
        (
            {"SeriesNumber": 3, "ImageType": ["ORIGINAL", "PRIMARY"]},
            ["SeriesNumber", "ImageType"],
            "3_ORIGINAL_PRIMARY",
        ),
        (
            {"SeriesDescription": "A" * 200},
            ["SeriesDescription"],
            "A" * util._MAX_INFO_ELEMENT_LENGTH,
        ),
    ],
)
def test_process_info_element(
    tags: dict[str, Any], infos: list[str], expected: str
) -> None:
    assert util._process_info_element(tags, infos) == expected


@pytest.mark.parametrize(
    ("patient_info", "expected"),
    [
        (False, "/BOA/20240101_ACC1_CT/3_Thorax/"),
        (True, "/BOA/Doe_19700101/20240101_ACC1_CT/3_Thorax/"),
    ],
)
def test_get_naming_scheme(patient_info: bool, expected: str) -> None:
    assert util.get_naming_scheme(TAGS, patient_info=patient_info) == expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"ORTHANC_USERNAME": "u", "ORTHANC_PASSWORD": "p"}, ("u", "p")),
        ({"ORTHANC__REGISTERED_USERS": '{"alice": "secret"}'}, ("alice", "secret")),
    ],
    indirect=["env"],
)
@pytest.mark.usefixtures("env")
def test_collect_auth(expected: tuple[str, str]) -> None:
    assert util.collect_auth() == expected


@pytest.mark.usefixtures("env")
def test_collect_auth_missing_raises() -> None:
    with pytest.raises(ValueError):
        util.collect_auth()
