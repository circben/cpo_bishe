import os


def test_project_has_data_raw_dir():
    assert os.path.isdir("data/raw")


def test_scripts_exist():
    assert os.path.isfile("scripts/stage1/download_datasets.py")
    assert os.path.isfile("scripts/stage2/build_preferences.py")
